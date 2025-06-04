from NewsScraping.items import NewsArticleItem
import scrapy


class TimeSpider(scrapy.Spider):
    name = "timespider"
    allowed_domains = ["time.com"]
    start_urls = ["https://time.com/section/politics/",
                  "https://time.com/section/world/",
                  "https://time.com/section/us/"
                 ]

    def parse(self, response):
        # Placeholder for selecting all articles on the page
        articles = response.css('.taxonomy-tout')  # Adjust the selector later
        
        for article in articles:

            # Extract relative URL for each article
           # if article.xpath('//a[contains(text(), "VIDEO")]').get():
            # continue  # Skip this article
            article_url = article.css('.taxonomy-tout a::attr(href)').get()
            
            
                
            # Follow each article link and scrape details from the article page
            yield response.follow(article_url, callback=self.parse_article_page)

        show_more_clicks = 0
        show_more_button_selector = '.selected-related__pages a.current + a'

        while show_more_clicks < 5:
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
        article_item['date'] = response.css('time:nth-of-type(1) ::text').get()
        article_item['publication'] = 'Time Magazine'  # Static value since all articles are from Fox News
        article_item['biasness'] = 'central'  # Static value since all articles are from Fox News
        article_item['score'] = '0.0'  # Static value since all articles are from Fox News
        

        # Yield the item to store or export it
        yield article_item
