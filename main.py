from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
import asyncio
import os
from typing import Optional, Dict, Any
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import httpx
import json

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for database
db_client = None
db = None

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')  # Optional: Get from https://huggingface.co/settings/tokens
if not HF_TOKEN:
    logger.error("HUGGINGFACE_TOKEN not found in environment variables!")
else:
    logger.info("HUGGINGFACE_TOKEN loaded successfully")

# Pydantic models for request/response
class ArticleMatchRequest(BaseModel):
    article_id: str

class ArticleInfo(BaseModel):
    id: str
    title: str
    summary: str

class MatchResponse(BaseModel):
    success: bool
    matched_article_id: Optional[str] = None
    similarity_score: Optional[float] = None
    source_article: Optional[ArticleInfo] = None
    matched_article: Optional[ArticleInfo] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None

class BackgroundProcessResponse(BaseModel):
    success: bool
    processed_count: int
    updated_count: int
    set_to_not_count: int
    message: str

# Database initialization function for serverless
async def init_database():
    global db_client, db
    
    if db_client is None:
        # Connect to MongoDB
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        try:
            db_client = AsyncIOMotorClient(mongodb_uri)
            db = db_client['Scraped-Articles-11']
            # Test connection
            await db_client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

# Connect to database on startup (for non-serverless)
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_database()
    yield
    
    # Cleanup
    if db_client:
        db_client.close()
        logger.info("MongoDB connection closed")

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Article Matching API",
    description="API to find semantically similar articles using Hugging Face API",
    version="1.0.0",
    lifespan=lifespan
)

async def get_embeddings_hf_api(texts: list) -> Optional[list]:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Update payload format to match the model's requirements
    payload = {
        "inputs": {
            "source_sentence": texts[0],
            "sentences": texts[1:] if len(texts) > 1 else [texts[0]]
        }
    }

    try:
        logger.info(f"Making request to HF API with {len(texts)} texts")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            logger.info(f"HF API Response Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                # The model returns similarity scores directly
                if isinstance(result, list):
                    return result
                else:
                    logger.error(f"Unexpected format from HF API: {result}")
            else:
                logger.error(f"HF API error: {response.status_code} - {response.text}")
                logger.error(f"Request URL: {HF_API_URL}")
                logger.error(f"Request Headers: {headers}")
    except Exception as e:
        logger.error(f"Error calling HF API: {e}")

    return None

def cosine_similarity_manual(vec1, vec2):
    """
    Calculate cosine similarity between two vectors manually
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

async def calculate_semantic_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    try:
        # Get similarity scores directly from the API
        similarities = await get_embeddings_hf_api([text1, text2])
        if similarities and len(similarities) > 0:
            # The API returns similarity scores directly
            return float(similarities[0])
        else:
            logger.warning("Failed to get valid similarity scores from HF API")
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

async def is_semantic_match(title1: str, summary1: str, title2: str, summary2: str, 
                           title_threshold: float = 0.6, summary_threshold: float = 0.5) -> tuple[bool, float]:
    """
    Determine if two articles are semantically similar based on title and summary
    Returns (is_match, combined_similarity_score)
    """
    title_similarity = await calculate_semantic_similarity(title1, title2)
    summary_similarity = await calculate_semantic_similarity(summary1, summary2)
    
    # Weighted combination (title has slightly more weight)
    combined_score = (title_similarity * 0.6) + (summary_similarity * 0.4)
    
    # Check if both title and summary meet their respective thresholds
    is_match = (title_similarity >= title_threshold or 
                summary_similarity >= summary_threshold)
    
    logger.info(f"Title similarity: {title_similarity:.3f}, Summary similarity: {summary_similarity:.3f}, Combined: {combined_score:.3f}")
    
    return is_match, combined_score


async def check_existing_backgrounds():
    """Check if any categories already have Background != 'None'"""
    categorized_collection = db['categorizedarticles']
    existing_with_background = []
    
    async for article in categorized_collection.find({"Background": {"$ne": "None"}}):
        existing_with_background.append(article)
    
    return existing_with_background


async def find_matching_background_for_category(category_doc):
    """Find matching background for a category using semantic similarity"""
    try:
        if not category_doc.get("articles"):
            return None
            
        # Get the first article from the category
        first_article_id = category_doc["articles"][0]
        
        # Find the article in Articles collection
        articles_collection = db['Articles']
        source_article = None
        
        try:
            from bson import ObjectId
            if ObjectId.is_valid(str(first_article_id)):
                source_article = await articles_collection.find_one({"_id": ObjectId(str(first_article_id))})
        except:
            pass
        
        if not source_article:
            source_article = await articles_collection.find_one({"_id": str(first_article_id)})
        
        if not source_article:
            return None
            
        source_title = source_article.get('title', '')
        source_summary = source_article.get('summary', '')
        
        if not source_title and not source_summary:
            return None
        
        # Get all categories with backgrounds
        categorized_collection = db['categorizedarticles']
        categories_with_background = []
        
        async for article in categorized_collection.find({"Background": {"$ne": "None"}}):
            if article.get('title', '').strip() or article.get('summary', '').strip():
                categories_with_background.append(article)
        
        if not categories_with_background:
            return None
        
        # Find the best semantic match
        best_match = None
        best_score = 0.0
        
        for category in categories_with_background:
            category_title = category.get('title', '')
            category_summary = category.get('summary', '')
            
            is_match, combined_score = await is_semantic_match(
                source_title, source_summary,
                category_title, category_summary
            )
            
            if is_match and combined_score > best_score:
                best_match = category
                best_score = combined_score
        
        if best_match:
            return str(best_match.get("_id"))  # Return the ID of the matching category
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding matching background: {e}")
        return None


async def process_background_assignments():
    """Process background assignments for categories with Background='None'"""
    categorized_collection = db['categorizedarticles']
    
    # Find all categories with Background='None'
    categories_to_process = []
    async for category in categorized_collection.find({"Background": "None"}):
        categories_to_process.append(category)
    
    if not categories_to_process:
        return {
            "processed_count": 0,
            "updated_count": 0,
            "set_to_not_count": 0,
            "message": "No categories with Background='None' found"
        }
    
    # Check if any existing categories have backgrounds
    existing_with_background = await check_existing_backgrounds()
    
    updated_count = 0
    set_to_not_count = 0
    
    if not existing_with_background:
        # No existing backgrounds, set all to "Not"
        result = await categorized_collection.update_many(
            {"Background": "None"},
            {"$set": {"Background": "Not"}}
        )
        set_to_not_count = result.modified_count
        logger.info(f"Set {set_to_not_count} categories to Background='Not' (no existing backgrounds)")
    else:
        # Try to match each category
        logger.info(f"Found {len(existing_with_background)} existing categories with backgrounds")
        
        for category in categories_to_process:
            matched_category_id = await find_matching_background_for_category(category)
            
            if matched_category_id:
                # Update with the ID of the matching category
                await categorized_collection.update_one(
                    {"_id": category["_id"]},
                    {"$set": {"Background": matched_category_id}}
                )
                updated_count += 1
                logger.info(f"Matched category {category['_id']} with category ID: {matched_category_id}")
            else:
                # No match found, set to "Not"
                await categorized_collection.update_one(
                    {"_id": category["_id"]},
                    {"$set": {"Background": "Not"}}
                )
                set_to_not_count += 1
                logger.info(f"No match for category {category['_id']}, set to 'Not'")
    
    return {
        "processed_count": len(categories_to_process),
        "updated_count": updated_count,
        "set_to_not_count": set_to_not_count,
        "message": f"Processed {len(categories_to_process)} categories"
    }


@app.post("/process-backgrounds", response_model=BackgroundProcessResponse)
async def process_backgrounds():
    """
    Process background assignments for all categories with Background='None'
    """
    try:
        # Initialize database connection for serverless
        await init_database()
        
        result = await process_background_assignments()
        
        return BackgroundProcessResponse(
            success=True,
            processed_count=result["processed_count"],
            updated_count=result["updated_count"],
            set_to_not_count=result["set_to_not_count"],
            message=result["message"]
        )
        
    except Exception as e:
        logger.error(f"Error processing backgrounds: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing backgrounds: {str(e)}"
        )


@app.post("/find-matching-article", response_model=MatchResponse)
async def find_matching_article(request: ArticleMatchRequest):
    """
    Find semantically similar article based on article ID
    """
    try:
        article_id = request.article_id
        
        # Step 1: Find the source article in "Articles" collection
        articles_collection = db['Articles']
        
        # Try to find by ObjectId first, then by string ID
        source_article = None
        try:
            from bson import ObjectId
            if ObjectId.is_valid(article_id):
                source_article = await articles_collection.find_one({"_id": ObjectId(article_id)})
        except:
            pass
        
        # If not found with ObjectId, try with string ID
        if not source_article:
            source_article = await articles_collection.find_one({"_id": article_id})
        
        if not source_article:
            raise HTTPException(
                status_code=404,
                detail="Article not found in Articles collection"
            )
        
        source_title = source_article.get('title', '')
        source_summary = source_article.get('summary', '')
        
        if not source_title and not source_summary:
            raise HTTPException(
                status_code=400,
                detail="Source article has no title or summary to compare"
            )
        
        # Step 2: Get all articles from "categorizedarticles" collection
        categorized_collection = db['categorizedarticles']
        categorized_articles = []
        
        async for article in categorized_collection.find({}):
            if article.get('title', '').strip() or article.get('summary', '').strip():
               categorized_articles.append(article)

        
        if not categorized_articles:
            return MatchResponse(
                success=True,
                matched_article_id=None,
                message="No articles found in categorizedarticles collection",
                source_article=ArticleInfo(
                    id=str(source_article['_id']),
                    title=source_title,
                    summary=source_summary
                )
            )
        
        # Step 3: Find the best semantic match
        best_match = None
        best_score = 0.0
        
        logger.info(f"Comparing against {len(categorized_articles)} categorized articles")
        
        for i, article in enumerate(categorized_articles):
            article_title = article.get('title', '')
            article_summary = article.get('summary', '')
            
            logger.info(f"Processing article {i+1}/{len(categorized_articles)}")
            
            is_match, combined_score = await is_semantic_match(
                source_title, source_summary,
                article_title, article_summary
            )
            
            if is_match and combined_score > best_score:
                best_match = article
                best_score = combined_score
                logger.info(f"New best match found with score: {best_score:.3f}")
        
        # Return the best match if found
        if best_match:
            return MatchResponse(
                success=True,
                matched_article_id=str(best_match['_id']),
                similarity_score=round(best_score, 4),
                source_article=ArticleInfo(
                    id=str(source_article['_id']),
                    title=source_title,
                    summary=source_summary
                ),
                matched_article=ArticleInfo(
                    id=str(best_match['_id']),
                    title=best_match.get('title', ''),
                    summary=best_match.get('summary', '')
                )
            )
        else:
            return MatchResponse(
                success=True,
                matched_article_id=None,
                message="No semantically similar article found",
                source_article=ArticleInfo(
                    id=str(source_article['_id']),
                    title=source_title,
                    summary=source_summary
                )
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding matching article: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/find-matching-article/{article_id}", response_model=MatchResponse)
async def find_matching_article_get(article_id: str):
    """
    Find semantically similar article using GET request with article ID in URL
    """
    request = ArticleMatchRequest(article_id=article_id)
    return await find_matching_article(request)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Initialize database connection for serverless
        await init_database()
        
        # Test database connection
        await db_client.admin.command('ping')
        
        # Test HF API token
        hf_status = "configured" if HF_TOKEN else "not configured"
        
        return {
            "status": "OK",
            "database": "connected",
            "huggingface_api": hf_status,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/similarity-test")
async def test_similarity(text1: str, text2: str):
    """
    Test endpoint to check semantic similarity between two texts
    """
    try:
        similarity = await calculate_semantic_similarity(text1, text2)
        return {
            "text1": text1,
            "text2": text2,
            "similarity_score": round(similarity, 4),
            "interpretation": "Very similar" if similarity > 0.8 else 
                           "Similar" if similarity > 0.6 else 
                           "Somewhat similar" if similarity > 0.4 else 
                           "Different"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating similarity: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )