from NewsScraping.items import NewsArticleItem
import scrapy


class FoxSpider(scrapy.Spider):
    name = "foxspider"
    allowed_domains = ["foxnews.com"]
    start_urls = ["https://foxnews.com/category/politics/executive/",
                  "https://foxnews.com/category/politics/senate/",
                  "https://foxnews.com/category/politics/house-of-representatives/",
                  "https://foxnews.com/category/politics/judiciary/",
                  "https://foxnews.com/category/politics/foreign-policy/",
                  "https://foxnews.com/category/politics/official-polls/",
                  "https://foxnews.com/category/politics/elections/",
                  "https://foxnews.com/category/us/crime/",
                  "https://foxnews.com/category/us/military/",
                  "https://foxnews.com/category/us/terror/",
                  "https://foxnews.com/category/us/immigration/",
                  "https://foxnews.com/category/us/economy/",
                  "https://foxnews.com/category/us/personal-freedoms/",
                  "https://foxnews.com/category/world/united-nations//",
                  "https://foxnews.com/category/world/conflicts/",
                  "https://foxnews.com/category/world/terrorism/",
                  "https://foxnews.com/category/world/disasters/",
                  "https://foxnews.com/category/world/global-economy/",
                  "https://foxnews.com/category/world/environment"]

    def parse(self, response):
        # Placeholder for selecting all articles on the page
        articles = response.css('article.article')  # Adjust the selector later
        
        for article in articles:

            # Extract relative URL for each article
           # if article.xpath('//a[contains(text(), "VIDEO")]').get():
            # continue  # Skip this article
            article_url = article.css('.article > .m a::attr(href)').get()
            
            
                
            # Follow each article link and scrape details from the article page
            yield response.follow(article_url, callback=self.parse_article_page)

        show_more_clicks = 0
        show_more_button_selector = '//*/footer/div/a'

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
        article_item['content'] = response.css('.article-body p ::text').getall()
        article_item['date'] = response.css('.article-date time ::text').get()
        article_item['publication'] = 'Fox News'  # Static value since all articles are from Fox News
        article_item['biasness'] = 'central'  # Static value since all articles are from Fox News
        article_item['score'] = '0.0'  # Static value since all articles are from Fox News
        

        # Yield the item to store or export it
        yield article_item
