from NewsScraping.items import NewsArticleItem
import scrapy


class guardianSpider(scrapy.Spider):
    name = "guardianspider"
    allowed_domains = ["theguardian.com"]
    start_urls = ["https://www.theguardian.com/world/africa/",
                  "https://www.theguardian.com/world/americas/",
                  "https://www.theguardian.com/world/asia-pacific/",
                  "https://www.theguardian.com/australia-news/australian-politics/",
                  "https://www.theguardian.com/world/europe-news/all/",
                  "https://www.theguardian.com/world/middleeast/",
                  "https://www.theguardian.com/world/south-and-central-asia/",
                  "https://www.theguardian.com/us-news/us-politics/",
                  "https://www.theguardian.com/politics/all/",
                  "https://www.theguardian.com/world/middleeast/",
                  "https://www.theguardian.com/world/ukraine/"]

    def parse(self, response):
        # Placeholder for selecting all articles on the page
        articles = response.css('li')  # Adjust the selector later
        
        for article in articles:

            # Extract relative URL for each article
           # if article.xpath('//a[contains(text(), "VIDEO")]').get():
            # continue  # Skip this article
            article_url = article.css('a::attr(href)').get()
            
            
                
            # Follow each article link and scrape details from the article page
            yield response.follow(article_url, callback=self.parse_article_page)

        show_more_clicks = 0
        show_more_button_selector = 'main section:last-child div:last-child div:last-child div:last-child a:last-child'

        while show_more_clicks < 200:
            show_more_button = response.css(show_more_button_selector)
            if show_more_button and show_more_button.xpath('@href').get():
                show_more_url = show_more_button.xpath('@href').get()
                yield response.follow(show_more_url, callback=self.parse)
                show_more_clicks += 1
            else:
                break  # No more "show more" button found

    def parse_article_page(self, response):
        # Create a NewsArticleItem object to store scraped data
        article_item = NewsArticleItem()

        # Filling up the article fields with placeholders for now
        article_item['title'] = response.css('h1 ::text').get()
        article_item['url'] = response.url
        article_item['content'] = response.css('p ::text').getall()
        article_item['date'] = response.css('summary span::text').get()

        article_item['publication'] = 'The Guardian'  # Static value since all articles are from the same source
        article_item['biasness'] = 'central'  # Static value since all articles are from Fox News
        article_item['score'] = '0.0'  # Static value since all articles are from Fox News
        

        # Yield the item to store or export it
        yield article_item
