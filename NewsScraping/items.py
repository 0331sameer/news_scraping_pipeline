# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import random
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

    if article.get('publication') == 'Breitbart':
        # List of image URLs to randomly select from 
        imageurls = [ 
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWwAAACKCAMAAAC5K4CgAAAAw1BMVEX///8AAAADBQT9/fvxVyO7u7v5vrLwTQfxVB3ybUjwTAD4xb1gYWDv7+/y8vLNzc2ZmZk4ODjg4OALDAuSkpKioqItLi5VVVXzgGP75uJra2vS0tIiIiL78/DxYThCQkLxZEDCwsK3t7dNTU3Pz8/Z2dmqqqpmZmZJSUmJiYkYGBiBgYF5eXnxURXn5+dZWVnwQADzinPzf2Lydlf40sn2pJMUFRX1m4f3s6T65OD52tT4zsbxWy7vNwD2rZ8pKirzjndCHviCAAAI00lEQVR4nO2bDVvayBqGZ0AqRRL5ECgIpUGDgkDssVVXe9b9/79q349MEiRQOK1mz17PfV1bSDKZxDsz77wzZI0BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8E/j5MN+9Iu+0X8Dn2vH+3B7UvSN/hv4WD3ahxpk/wYg+x2B7HdEZFfzgOzfjsj+tMF/7j7+qNWW1TXZ3qQiBC05tVyJN0OtKqgknBqfy5rWJN1XCUwYb56ukuvLSRPdDt3ZHm95lSyu/MQV9+kzNOXMBUKubMLnriZ8C8QpX5VodQb1xTS9akGQ7OqXnP3l8qj/7fvHWjUj+9Q6ri9p0082mxM+5SXZtpFp8YeZ2AwvpuO+liIvvo5uN+T7LDlcp8Nh9lxrsuW5OF/gzHiZYjMzTf+yZL7smEvfW3uRqbjxZhr3Q2WXc6HDTx+Ps7JLiv5NJDvZ7NDhpjtcsnVyUbJt+qtLCbZJspMK2trMPNljB7Jxlh4ueSw7PTeVzVs9+rISeV6mzJlZ0EX5uU/ps8ul55arnnFN9F/htnfJFt+fluuypeWV7DyWHW9aT2VnWrbK1uMl17LjEonfS5V9IRtna4fD7Lnrsvlqqey0ZTfoog1XEd273+a9K/F8L//6RThO+Ilsiiafq1nZNqhMLvgzFNmWouOAN6ci2zZczFbZErOH9PVMYjbJtuOgMu3S51z+bt5jXcMlR/aCDr/IYZ9P4ToXQSZm68MJMrKtnbqYPeHSVEjuaKVlpvII5oEXcrXTQiQ7NmXz3uzmwzIjW7VYuW2RzcpIvr1R2aGrV2ULrFaGK1EbGQ7Hpbgo9fDu3IoZlT3QwzauqfdakMq+0Zgdy3bx3wS0daU3JPVTv6HncsUPm/ZW4kdRHPEAmWnJ379/f/wj3WFOahuy+fY7qexYEss+d/Vulc3xtpwUpSGtPnDHnexyN6mpJ1fKoLJf1lp2IpsjD0ekMcs+Vb2h1DEx+niiN5G4L5z6/fj6dZTI7d8eL3kpJJH9vCl7ILExkd04VDbb0KI0gi0aTqiTnR7Olc2X9XJle20eGeRZSoeYUplV0jtWw2az9yYS90Xy7OWPVPbolnd8SmU/HufK7vwO2Zy5dZLuvZ9sO55zJM6TXR7KHXlzK8kJVWhffOmHUn/5TQwegM4gP7+Sffw1lf2pmhtGKr8gm5qetTwzkqDKIfqaD+8pm+IOFcuTbYYS/luSspDfhaTbfHu28AkNkyObo3g/cf1UO3otu8yJlrcue7avbI6a55azbqMJcehzO+StTMzWZ7FFNieU+bIjuYVzq7k+DZT8bNl9wcE6ZkN2uT/qj5IR0/SPqmt5tr08rYy1XyayI2noIvvS1btV9vjydDp0OcZMpi8ahFPZfG5Xz82THbHnMCM7bbU3cieB1TkU9RCpb5Hm9cWyKXstyX5U168nNfaqHMsOVy12RHMckd0cE83WdtnxrIV7guHHZJtlSR25AFdU91aXY1tyc71c2XQl2/HSSQ1fc3zNEVlnNdxfuA/6XZ0zagy/ehefO9kt+/nu9rVsSb2G7EZkqzoJH+kMMtwhO566a9dvSj9fxA09nUFalzXkye7RAG17mem6nHPPslnzDSm3/OCNd2+1A53LpLTYTITZLduYpx/VvJbNLcZN16kZSfTYT7Z7PLyHbdwkMV/XRta95MuuWDs8b+fIPpVAROMixxOP70EW/XgJjRcFik5HNmXz3szW6Mv6dH0y7UT82cqsjVj5m/YKI7Y5mTa4JOVkMlWc6sSP5yKuZdO/17GXfNncqjvznDCyksTmylpOJ1uc68RT2vDe6phZKJuyn799xmlZmBPm1P1SJqihJHB4Mqt8Ow3QHJa4Hd1SK1o0+OyQ6Mxu7tYyGRbY/oW2TzyRcOcAdKXO+xRLkNHgyBZXKRrDOOJTpFspn7/5R8N/szY5kR7XXYgqZvILss0WObDB0xqZhJbedLBKR4vjLe9NBvhz6E27S2yuUSptJn6yQSTRwL/2lIfpGiTLPOt0gWvwsif1Bzd9lPZH45fyw4lN0tSvxubLETtKXuqWqlTyOLfWPuEk+3Hcd9sla1DdY5svhYHGVOnQDOTybsjkCD4u/0dRP50/ej2KZ1C8krUTtkde+h0PZZNza8d9XrRULvG2gxSz9gi2+hC96bsOp9Jj5ATltlCl6UqQSArtNfaJgpkYyFqdLtcLmtfMkPkyc9a9vR/lJ3kjslSRka2phHbZPe2yOYsssH5JEWoK+o4vCbifnto2HhZoDA2lljLfz08PDxms5M/30p2K8kVdT59gOzOFtlcRcSOKV6/XFhZx3Z3PbHxIkFh5P94sPbrwd3GADn5Vdkc5RcyznYlV6Qv4/IhsltbZPOPnpIqcd2aWMp9uWhXbPL305/FzPOrJVbflx+YBrmyA1/ZIbvs+0FJPHC/lnHQ03TkANlmmMlGWu6iRteguO4w+f1MFoTP4nzz7D2cbuVnsuOVqHRSMxwOpdsHubKHwnywXXZbK+Ccjy1IqxQP4UGyB4lsd9E2i1w5xysdQnntUFp781oez7kpkt2yjTlZ5k/XL8y67Prar+v1HWsj8c/xM3WqRThRqBwku5LKdhdl2b7KDo3/IkckzY5vLPkVvzC2yeZj5dGHu1ruSzpu1U/fDajIewqmmx6uxy/pCPzyjqrrpCV4hkgf8YA1kD08x5G1OZoCxhnxhbWvZGs6JzN2kb1eJYUXRl6sSO6g5V4fGhf7JoPK7m/wx9Pz48Pd0r0QpbLDXl24akik9elrj29/Rfsj3yzqjmgm++LRaEDFtPtOIj0+6HD39nr1KI6hdIC+TqN4R4e+qOIb+lbJ3u9FPdJcma4WTY3fSy86jXfTV2oKN3KnepLfoCDy0pv+IxaiNt6sPFrWasfLzLuV//cvVur4WTB4Zfgdgex3BLLfEfwPTO8I/tc8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAXzN3cJxgeqLQCCAAAAAElFTkSuQmCC",
            "data:image/png;base64,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",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsVtIpy9szi2ggZs8WF8fwI5_8gOfciXXMNg&s",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxHKj5g7qpy5d7i1McHHRC4Bw5fKjYFzLRSg&s" 
            ]
        image_url = random.choice(imageurls)
    
    # Return processed article
    return {
        **article,
        'content': content,
        'biasness': bias,
        'score': score,
        'date': date,
        'image_url': image_url,
    }