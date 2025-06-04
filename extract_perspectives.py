import pymongo
import requests
import time

# MongoDB connection
MONGO_URI = (
    "mongodb+srv://jamshidjunaid763:JUNAID12345@insightwirecluster.qz5cz."
    "mongodb.net/?retryWrites=true&w=majority&appName=InsightWireCluster"
)
client = pymongo.MongoClient(MONGO_URI)
db = client["Scraped-Articles-11"]
articles_collection = db["Articles"]
perspectives_collection = db["perspectives"]

PERSPECTIVE_API_URL = "https://story-perspectives.vercel.app/rewrite"


def fetch_articles_for_perspective():
    return list(
        articles_collection.find({"articlesperscpectives": False})
    )

def get_perspective(article_id, bias_tag):
    payload = {
        "article_id": str(article_id),
        "bias_tag": bias_tag
    }
    try:
        response = requests.post(PERSPECTIVE_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API error for article {article_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Exception for article {article_id}: {e}")
        return None

def main():
    articles = fetch_articles_for_perspective()
    print(f"Found {len(articles)} articles needing perspective.")
    processed_ids = []
    for article in articles:
        article_id = article["_id"]
        bias_tag = article.get("biasness", "center")
        print(f"Processing article {article_id} with bias '{bias_tag}'...")
        perspective = get_perspective(article_id, bias_tag)
        if perspective:
            perspectives_collection.insert_one(perspective)
            processed_ids.append(article_id)
            print(f"✅ Perspective stored for article {article_id}.")
        time.sleep(1)  # Be polite to the API
    if processed_ids:
        articles_collection.update_many(
            {"_id": {"$in": processed_ids}},
            {"$set": {"articlesperscpectives": True}}
        )
        print(f"✅ Updated {len(processed_ids)} articles as perspective processed.")
    else:
        print("No articles were processed.")

if __name__ == "__main__":
    main() 