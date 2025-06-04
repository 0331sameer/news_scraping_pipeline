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
    print("Fetching articles from MongoDB...")
    articles = list(
        articles_collection.find(
            {"artcles_categorized": False}
        )
    )
    print(f"✅ Fetched {len(articles)} articles.")
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
    print("Clustering articles...")
    for article_id, entities in tqdm(
        article_signatures.items(), desc="Clustering"
    ):
        matched_cluster = None
        for cluster_id, cluster_articles in clusters.items():
            common_entities = sum(
                1
                for other_id in cluster_articles
                if len(entities & article_signatures[other_id]) >= 3
            )
            if common_entities >= len(cluster_articles) / 2:
                matched_cluster = cluster_id
                break
        if matched_cluster is not None:
            clusters[matched_cluster].append(article_id)
        else:
            clusters[len(clusters)] = [article_id]
    print(f"✅ Generated {len(clusters)} clusters.")
    return clusters


def generate_summary(texts, summary_pipeline):
    texts = [" ".join(text) if isinstance(text, list) else text for text in texts]
    combined_text = " ".join(texts[:5])
    summary = summary_pipeline(
        combined_text[:1024],
        max_length=150,
        min_length=50,
        do_sample=False,
        truncation=True,
    )[0]["summary_text"]
    return summary


def generate_topic(summary_text, topic_pipeline):
    topic_prompt = (
        "Generate a one phrase news title from the summary. "
        "it should say what happened, who did it and to whom. "
        "mentioned the names of the entities involved.: "
        f"{summary_text}"
    )
    initial_topic = topic_pipeline(
        topic_prompt, max_new_tokens=25, do_sample=False, truncation=True
    )[0]["generated_text"]
    topic = initial_topic.replace(topic_prompt, "").strip()
    return topic


def main():
    articles = fetch_articles()
    if not articles:
        print("No articles found in the database. Exiting.")
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
    clusters = cluster_articles(article_signatures, article_relationships)
    print("Loading AI models for summarization & topic generation...")
    summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    topic_pipeline = pipeline("text-generation", model="openai-community/gpt2")
    print("✅ AI models loaded.")
    cluster_documents = []
    print("Generating cluster documents...")
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
        topic = generate_topic(summary, topic_pipeline)
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
    if cluster_documents:
        clusters_collection.drop()
        clusters_collection.insert_many(cluster_documents)
        print("✅ Clusters stored successfully in MongoDB!")
        # Mark processed articles as categorized
        all_article_ids = [aid for cluster in cluster_documents for aid in cluster["articles"]]
        if all_article_ids:
            articles_collection.update_many(
                {"_id": {"$in": all_article_ids}},
                {"$set": {"artcles_categorized": True}}
            )
            print(f"✅ Marked {len(all_article_ids)} articles as categorized.")
    else:
        print("No clusters to store.")


if __name__ == "__main__":
    main() 