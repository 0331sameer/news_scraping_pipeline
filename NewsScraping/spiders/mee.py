from NewsScraping.items import NewsArticleItem
import scrapy


class meeSpider(scrapy.Spider):
    name = "meespider"
    allowed_domains = ["middleeasteye.net"]
    start_urls = ["https://www.middleeasteye.net/news/",
                  "https://www.middleeasteye.net/trending/",
                  "https://www.middleeasteye.net/opinion/"]

    def parse(self, response):
        # Placeholder for selecting all articles on the page
        articles = response.css('.mee-tile-image')  # Adjust the selector later
        
        for article in articles:

            # Extract relative URL for each article
           # if article.xpath('//a[contains(text(), "VIDEO")]').get():
            # continue  # Skip this article
            article_url = article.css('a::attr(href)').get()
            
            
                
            # Follow each article link and scrape details from the article page
            yield response.follow(article_url, callback=self.parse_article_page)

        show_more_clicks = 0
        show_more_button_selector = '.pager__item.close-page a'

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
        article_item['title'] = response.css('.field.field-title ::text').get()
        article_item['url'] = response.url
        article_item['content'] = response.css('p ::text').getall()
        article_item['date'] = response.css('.date-created').xpath('following-sibling::text()').get().strip()

        article_item['publication'] = 'Middle East Eye'  # Static value since all articles are from Fox News
        article_item['biasness'] = 'central'  # Static value since all articles are from Fox News
        article_item['score'] = '0.0'  # Static value since all articles are from Fox News

        # Yield the item to store or export it
        yield article_item
