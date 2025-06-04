# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
import re
from datetime import datetime
from dateutil import parser as date_parser
from transformers import pipeline
import requests
from bs4 import BeautifulSoup



class NewsArticleItem(scrapy.Item):
    id = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    date = scrapy.Field()
    author = scrapy.Field()
    imageUrl = scrapy.Field()
    publication = scrapy.Field()
    category = scrapy.Field()
    url = scrapy.Field()
    location = scrapy.Field()
    date = scrapy.Field()  # Ensure this line is present
    biasness = scrapy.Field(default="central")  # Added biasness field with default value
    score = scrapy.Field(default="0")  # Added biasness field with default value

# Initialize the classifier once
model_name = "sameer35/distilbert-political-bias"
classifier = pipeline("text-classification", model=model_name, token="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")

def get_bias_and_score(text):
    result = classifier(text)[0]
    label_map = {'LABEL_0': 'right', 'LABEL_1': 'center', 'LABEL_2': 'left'}
    label = label_map.get(result['label'], 'unknown')
    score = float(result['score'])
    return label, score

def clean_content(content):
    if isinstance(content, list):
        content = ' '.join(content)
    # Remove tabs, newlines, and excessive whitespace
    content = re.sub(r'[\t\n\r]+', ' ', content)
    content = re.sub(r'\s+', ' ', content).strip()
    return content

def normalize_date(date_str):
    try:
        # Try parsing with dateutil
        dt = date_parser.parse(date_str, fuzzy=True, dayfirst=False)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        # If parsing fails, use current date
        return datetime.now().strftime('%Y-%m-%d')

def extract_image_url(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return "https://via.placeholder.com/800x450?text=No+Image+Available"
        soup = BeautifulSoup(resp.text, "html.parser")
        # OpenGraph
        image_url = soup.find("meta", property="og:image")
        if image_url and image_url.get("content"):
            return image_url["content"]
        # Twitter
        image_url = soup.find("meta", attrs={"name": "twitter:image"})
        if image_url and image_url.get("content"):
            return image_url["content"]
        # First large image
        for img in soup.find_all("img"):
            src = img.get("src")
            width = img.get("width")
            height = img.get("height")
            if src and (
                (width and height and int(width) >= 200 and int(height) >= 200) or
                (not width and not height and any(x in src for x in ["jpg", "jpeg", "png"]))
            ):
                if src.startswith("http"):
                    return src
                elif src.startswith("/"):
                    base_url = requests.utils.urlparse(url)
                    return f"{base_url.scheme}://{base_url.netloc}{src}"
        # Fallback: any image
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and any(x in src for x in ["jpg", "jpeg", "png"]):
                if src.startswith("http"):
                    return src
                elif src.startswith("/"):
                    base_url = requests.utils.urlparse(url)
                    return f"{base_url.scheme}://{base_url.netloc}{src}"
        return "https://via.placeholder.com/800x450?text=No+Image+Available"
    except Exception:
        return "https://via.placeholder.com/800x450?text=Error+Loading+Image"

def process_article(article):
    # Clean and join content
    content = clean_content(article.get('content', ''))
    # Get bias and score
    bias, score = get_bias_and_score(content)
    # Normalize date
    date = normalize_date(article.get('date', ''))
    # Get image url
    image_url = extract_image_url(article.get('url', ''))
    # Return processed article
    return {
        **article,
        'content': content,
        'biasness': bias,
        'score': score,
        'date': date,
        'image_url': image_url,
    }