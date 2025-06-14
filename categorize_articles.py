import spacy
import pymongo
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from NewsScraping.items import extract_image_url

# Load SpaCy model
print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_trf")

# Connect to MongoDB (Scraped-Articles-11)
MONGO_URI = (
    "mongodb+srv://jamshidjunaid763:JUNAID12345@insightwirecluster.qz5cz."
    "mongodb.net/?retryWrites=true&w=majority&appName=InsightWireCluster"
)
client = pymongo.MongoClient(MONGO_URI)
db = client["Scraped-Articles-11"]
articles_collection = db["Articles"]
clusters_collection = db["categorizedarticles"]
print("Connected to MongoDB!")


def fetch_articles():
    print("Fetching uncategorized articles from MongoDB...")
    articles = list(
        articles_collection.find(
            {"artcles_categorized": {"$ne": True}}
        )
    )
    print(f"✅ Fetched {len(articles)} uncategorized articles.")
    return articles


def extract_event_signature(article_text):
    if isinstance(article_text, list):
        article_text = " ".join(article_text)
    doc = nlp(article_text)
    entities = [
        ent.text.lower()
        for ent in doc.ents
        if ent.label_ in ("EVENT", "ORG", "GPE", "DATE", "PERSON")
    ]
    entity_pairs = []
    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "pobj", "appos"):
            head = token.head.text.lower()
            if head in entities and token.text.lower() in entities:
                entity_pairs.append((head, token.text.lower()))
    return set(entities), entity_pairs


def compute_tfidf_weighted_entities(articles, article_signatures):
    corpus = [" ".join(article_signatures[a["_id"]]) for a in articles]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    entity_scores = {}
    feature_names = vectorizer.get_feature_names_out()
    for i, article in enumerate(articles):
        article_id = article["_id"]
        scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
        entity_scores[article_id] = {
            entity: score for entity, score in scores if score > 0
        }
    return entity_scores


def cluster_articles(article_signatures, article_relationships):
    clusters = defaultdict(list)
    unclustered_articles = []
    print("Clustering articles...")
    
    for article_id, entities in tqdm(
        article_signatures.items(), desc="Clustering"
    ):
        matched_cluster = None
        best_similarity = 0
        
        # Find the best matching cluster
        for cluster_id, cluster_articles in clusters.items():
            if not cluster_articles:  # Skip empty clusters
                continue
                
            # Calculate similarity with existing cluster
            cluster_similarities = []
            for other_id in cluster_articles:
                other_entities = article_signatures[other_id]
                if len(entities) == 0 and len(other_entities) == 0:
                    similarity = 0
                else:
                    # Use Jaccard similarity: intersection / union
                    intersection = len(entities & other_entities)
                    union = len(entities | other_entities)
                    similarity = intersection / union if union > 0 else 0
                cluster_similarities.append(similarity)
            
            # Average similarity with cluster
            avg_similarity = sum(cluster_similarities) / len(cluster_similarities)
            
            # Require minimum similarity threshold and minimum shared entities
            min_shared_entities = min(2, max(1, len(entities) // 3))  # Adaptive threshold
            shared_entities = sum(
                1 for other_id in cluster_articles
                if len(entities & article_signatures[other_id]) >= min_shared_entities
            )
            
            # Cluster criteria: good similarity AND sufficient entity overlap
            similarity_threshold = 0.3  # 30% similarity threshold
            min_shared_threshold = max(1, len(cluster_articles) // 2)
            if (avg_similarity >= similarity_threshold and
                shared_entities >= min_shared_threshold):
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    matched_cluster = cluster_id
        
        if matched_cluster is not None:
            clusters[matched_cluster].append(article_id)
        else:
            # Don't create single-article clusters immediately
            unclustered_articles.append(article_id)
    
    # Try to cluster remaining unclustered articles with each other
    print(f"Attempting to cluster {len(unclustered_articles)} remaining articles...")
    
    for article_id in unclustered_articles:
        entities = article_signatures[article_id]
        matched_cluster = None
        best_similarity = 0
        
        # Check if it can form a cluster with other unclustered articles
        for other_id in unclustered_articles:
            if article_id == other_id:
                continue
                
            other_entities = article_signatures[other_id]
            if len(entities) == 0 and len(other_entities) == 0:
                continue
                
            intersection = len(entities & other_entities)
            union = len(entities | other_entities)
            similarity = intersection / union if union > 0 else 0
            
            # Lower threshold for creating new clusters
            if similarity >= 0.25 and intersection >= 2:
                # Find existing cluster or create new one
                found_cluster = None
                for cluster_id, cluster_articles in clusters.items():
                    if other_id in cluster_articles:
                        found_cluster = cluster_id
                        break
                
                if found_cluster is not None:
                    if similarity > best_similarity:
                        best_similarity = similarity
                        matched_cluster = found_cluster
                else:
                    # Create new cluster with both articles
                    new_cluster_id = len(clusters)
                    clusters[new_cluster_id] = [other_id, article_id]
                    # Remove from unclustered list
                    if other_id in unclustered_articles:
                        unclustered_articles = [x for x in unclustered_articles if x != other_id]
                    break
        
        if matched_cluster is not None and article_id not in clusters[matched_cluster]:
            clusters[matched_cluster].append(article_id)
        elif not any(article_id in cluster_articles for cluster_articles in clusters.values()):
            # Only create single-article cluster if article has substantial content
            if len(entities) >= 3:  # Only if article has meaningful entities
                clusters[len(clusters)] = [article_id]
    
    # Filter out clusters that are too small (optional - keep for now but mark them)
    valid_clusters = {}
    single_article_clusters = {}
    
    for cluster_id, cluster_articles in clusters.items():
        if len(cluster_articles) >= 2:
            valid_clusters[cluster_id] = cluster_articles
        else:
            single_article_clusters[cluster_id] = cluster_articles
    
    print(f"✅ Generated {len(valid_clusters)} multi-article clusters and {len(single_article_clusters)} single-article clusters.")
    
    # Return both types but we'll handle them differently
    return valid_clusters, single_article_clusters


def generate_summary(texts, summary_pipeline):
    """Generate a comprehensive 4+ line summary from multiple article texts."""
    texts = [" ".join(text) if isinstance(text, list) else text for text in texts]
    
    # Use more text for better context (up to 8 articles, more characters)
    combined_text = " ".join(texts[:8])
    
    # Truncate to fit model limits but keep more content
    if len(combined_text) > 2048:
        combined_text = combined_text[:2048]
    
    # Generate longer, more detailed summary
    summary = summary_pipeline(
        combined_text,
        max_length=300,  # Increased from 150 to ensure 4+ lines
        min_length=120,  # Increased from 50 to ensure substantial content
        do_sample=True,  # Enable sampling for more varied output
        temperature=0.7,  # Add some creativity while maintaining coherence
        truncation=True,
    )[0]["summary_text"]
    
    # Post-process to ensure quality
    sentences = summary.split('. ')
    if len(sentences) < 4:
        # If summary is too short, try generating another one with different parameters
        extended_summary = summary_pipeline(
            combined_text,
            max_length=400,
            min_length=150,
            do_sample=True,
            temperature=0.8,
            truncation=True,
        )[0]["summary_text"]
        
        # Use the longer one
        if len(extended_summary.split('. ')) > len(sentences):
            summary = extended_summary
    
    return summary


def generate_topic(summary_text, topic_pipeline, cluster_articles_list):
    """Generate an informative news title that captures the main story."""
    
    # Extract key information from articles for better title generation
    titles = [article.get("title", "") for article in cluster_articles_list if article.get("title")]
    sources = list(set([article.get("source", "") for article in cluster_articles_list if article.get("source")]))
    
    # Create a more detailed prompt for better title generation
    title_prompt = (
        f"Write a clear, informative news headline based on this summary. "
        f"The headline should be specific, mention key people/organizations, and capture the main event. "
        f"Keep it under 15 words and make it engaging:\n\n"
        f"Summary: {summary_text[:500]}\n\n"
        f"Headline:"
    )
    
    try:
        # Generate title with better parameters
        generated_response = topic_pipeline(
            title_prompt, 
            max_new_tokens=30,  # Increased for longer titles
            do_sample=True,
            temperature=0.6,  # Balanced creativity
            pad_token_id=topic_pipeline.tokenizer.eos_token_id,
            num_return_sequences=1
    )[0]["generated_text"]
        
        # Extract just the headline part
        topic = generated_response.replace(title_prompt, "").strip()
        
        # Clean up the generated title
        topic = topic.split('\n')[0].strip()  # Take first line only
        topic = topic.replace('"', '').replace("'", "")  # Remove quotes
        
        # If title is too short or generic, create a fallback
        if len(topic.split()) < 4 or topic.lower().startswith(('the', 'a', 'an', 'this', 'that')):
            # Create fallback title from summary key points
            summary_words = summary_text.split()[:50]  # First 50 words
            key_phrases = []
            
            # Look for action words and entities
            action_words = ['announces', 'reports', 'says', 'confirms', 'reveals', 'launches', 'plans', 'agrees', 'decides']
            for i, word in enumerate(summary_words):
                if word.lower() in action_words and i > 0:
                    # Take context around action word
                    start = max(0, i-3)
                    end = min(len(summary_words), i+4)
                    key_phrases.append(' '.join(summary_words[start:end]))
                    break
            
            if key_phrases:
                topic = key_phrases[0].strip()
            else:
                # Last resort: use first meaningful sentence from summary
                sentences = summary_text.split('. ')
                topic = sentences[0] if sentences else "News Update"
        
        # Ensure title is not too long
        if len(topic) > 100:
            topic = topic[:97] + "..."
            
        return topic.strip()
        
    except Exception as e:
        print(f"Warning: Title generation failed ({e}), using fallback")
        # Fallback to first sentence of summary
        sentences = summary_text.split('. ')
        return sentences[0][:100] if sentences else "News Update"


def main():
    print("🚀 Starting Article Categorization Process")
    print("=" * 50)
    print("ℹ️  This process will:")
    print("   • Only process NEW uncategorized articles")
    print("   • PRESERVE existing categories in the database")
    print("   • Add new categories alongside existing ones")
    print("-" * 50)
    
    articles = fetch_articles()
    if not articles:
        print("✅ No new uncategorized articles found. All articles are already processed!")
        return
    article_signatures = {}
    article_relationships = {}
    print("Extracting event signatures & relationships...")
    for article in tqdm(articles, desc="Processing articles"):
        entities, relationships = extract_event_signature(article["content"])
        article_signatures[article["_id"]] = entities
        article_relationships[article["_id"]] = relationships
    print("✅ Event signatures & relationships extracted!")
    print("Computing TF-IDF weights...")
    compute_tfidf_weighted_entities(articles, article_signatures)
    print("✅ TF-IDF entity weighting completed!")
    clusters, single_article_clusters = cluster_articles(article_signatures, article_relationships)
    print("Loading AI models for summarization & topic generation...")
    summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    topic_pipeline = pipeline("text-generation", model="openai-community/gpt2")
    print("✅ AI models loaded.")
    cluster_documents = []
    print("Generating cluster documents...")
    
    # Only process multi-article clusters for categorization
    for cluster_id, article_ids in tqdm(
        clusters.items(), desc="Saving clusters"
    ):
        cluster_articles_list = [
            article for article in articles if article["_id"] in article_ids
        ]
        texts = [
            article["content"]
            for article in cluster_articles_list
        ]
        summary = generate_summary(texts, summary_pipeline)
        topic = generate_topic(summary, topic_pipeline, cluster_articles_list)
        # Get image URL of the top article (first in cluster)
        image_url = None
        if cluster_articles_list:
            top_article = cluster_articles_list[0]
            image_url = top_article.get("image_url")
        cluster_doc = {
            "title": topic,
            "summary": summary,
            "articles": article_ids,
            "image_url": image_url,
            "Background": "None",
            "Analytics": [],
        }
        cluster_documents.append(cluster_doc)
    
    # Store results
    if cluster_documents:
        # Insert new clusters without dropping existing ones
        clusters_collection.insert_many(cluster_documents)
        print("✅ New clusters stored successfully in MongoDB!")
        # Only mark articles as categorized if they're in multi-article clusters
        multi_article_ids = [aid for cluster in cluster_documents for aid in cluster["articles"]]
        if multi_article_ids:
            articles_collection.update_many(
                {"_id": {"$in": multi_article_ids}},
                {"$set": {"artcles_categorized": True}}
            )
            print(f"✅ Marked {len(multi_article_ids)} articles as categorized.")
        
        # Report on single-article clusters (these remain uncategorized)
        single_article_ids = [aid for cluster in single_article_clusters.values() for aid in cluster]
        if single_article_ids:
            print(f"ℹ️  {len(single_article_ids)} articles remain uncategorized (insufficient similarity).")
    else:
        print("No multi-article clusters to store.")
        print(f"ℹ️  Found {len(single_article_clusters)} single-article clusters - these remain uncategorized.")


if __name__ == "__main__":
    main() 