from NewsScraping.items import NewsArticleItem
import scrapy


class aljazeeraSpider(scrapy.Spider):
    name = "aljazeeraspider"
    allowed_domains = ["aljazeera.com"]
    start_urls = ["https://www.aljazeera.com/features/",
                  "https://www.aljazeera.com/tag/ukraine-russia-crisis/",
                  "https://www.aljazeera.com/economy/",
                  "https://www.aljazeera.com/investigations/",
                  "https://www.aljazeera.com/middle-east/",
                  "https://www.aljazeera.com/news/",
                  "https://www.aljazeera.com/africa/",
                  "https://www.aljazeera.com/asia/",
                  "https://www.aljazeera.com/us-canada/",
                  "https://www.aljazeera.com/latin-america/",
                  "https://www.aljazeera.com/europe/",
                  "https://www.aljazeera.com/asia-pacific/"]

    def parse(self, response):
        # Placeholder for selecting all articles on the page
        articles = response.css('article')  # Adjust the selector later
        
        for article in articles:

            # Extract relative URL for each article
           # if article.xpath('//a[contains(text(), "VIDEO")]').get():
            # continue  # Skip this article
            article_url = article.css('a::attr(href)').get()
            
            
                
            # Follow each article link and scrape details from the article page
            yield response.follow(article_url, callback=self.parse_article_page)

        

    def parse_article_page(self, response):
        # Create a NewsArticleItem object to store scraped data
        article_item = NewsArticleItem()

        # Filling up the article fields with placeholders for now
        article_item['title'] = response.css('h1 ::text').get()
        article_item['url'] = response.url
        article_item['content'] = response.css('p ::text').getall()
        article_item['date'] = response.css('.date-simple span:nth-of-type(2)::text').get()
        article_item['publication'] = 'Al Jazeera'  # Static value since all articles are from Fox News
        article_item['biasness'] = 'central'  # Static value since all articles are from Fox News
        article_item['score'] = '0.0'  # Static value since all articles are from Fox News
        

        # Yield the item to store or export it
        yield article_item
