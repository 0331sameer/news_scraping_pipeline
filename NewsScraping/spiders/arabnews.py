from NewsScraping.items import NewsArticleItem
import scrapy


class arabnewsSpider(scrapy.Spider):
    name = "arabnewsspider"
    allowed_domains = ["arabnews.com"]
    start_urls = ["https://www.arabnews.com/world/",
                  "https://www.arabnews.com/middleeast/"
                  
                 ]

    def parse(self, response):
        # Placeholder for selecting all articles on the page
        articles = response.css('.article-item')  # Adjust the selector later
        
        for article in articles:

            # Extract relative URL for each article
           # if article.xpath('//a[contains(text(), "VIDEO")]').get():
            # continue  # Skip this article
            article_url = article.css('.article-item a::attr(href)').get()
            
            
                
            # Follow each article link and scrape details from the article page
            yield response.follow(article_url, callback=self.parse_article_page)

       

    def parse_article_page(self, response):
        # Create a NewsArticleItem object to store scraped data
        article_item = NewsArticleItem()

        # Filling up the article fields with placeholders for now
        article_item['title'] = response.css('h1 ::text').get()
        article_item['url'] = response.url
        article_item['content'] = response.css('.field-items p ::text').getall()
        article_item['date'] = response.css('.entry-date  time ::text').get()
        print(article_item['date'])
        article_item['publication'] = 'Arab News'  # Static value since all articles are from Fox News
        article_item['biasness'] = 'central'  # Static value since all articles are from Fox News
        article_item['score'] = '0.0'  # Static value since all articles are from Fox News
        

        # Yield the item to store or export it
        yield article_item
