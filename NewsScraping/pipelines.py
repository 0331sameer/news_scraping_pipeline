# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import logging
import pymongo
from NewsScraping.items import process_article
# from itemadapter import ItemAdapter  # Unused import, commented out for lint


# MongoDB connection settings
MONGO_URI = (
    "mongodb+srv://jamshidjunaid763:JUNAID12345@insightwirecluster.qz5cz."
    "mongodb.net/?retryWrites=true&w=majority&appName=InsightWireCluster"
)
DB = 'Scraped-Articles-11'
COLLECTION = 'Articles'


class BasePipeline:
    def __init__(self):
        self.items = []
        self.articles_scraped = 0
        self.articles_skipped = 0

    def open_spider(self, spider):
        # Connect to MongoDB
        logging.info("Opening connection to MongoDB")
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client[DB]
        self.collection = self.db[COLLECTION]

        # Create an index on the 'title' field for faster search
        self.collection.create_index("title", unique=True)

        self.items = []
        self.articles_scraped = 0
        self.articles_skipped = 0
        print("MongoDB connected successfully!")

    def close_spider(self, spider):
        # Insert all collected items in bulk
        if self.items:
            try:
                self.collection.insert_many(self.items)
                logging.info(
                    f"Inserted {len(self.items)} items into the collection."
                )
            except Exception as e:
                logging.error(f"Failed to insert items: {e}")

        # Log final statistics
        logging.info(f"Spider {spider.name} statistics:")
        logging.info(f"Total articles scraped: {self.articles_scraped}")
        logging.info(f"Articles skipped (duplicates): {self.articles_skipped}")
        logging.info(f"Articles stored: {len(self.items)}")

        # Close the connection
        logging.info("Closing connection to MongoDB")
        self.client.close()

    def process_item(self, item, spider):
        try:
            # Process the item using process_article
            processed_item = process_article(dict(item))
            # Add default fields
            processed_item['articlesperscpectives'] = False
            processed_item['artcles_categorized'] = False
            # Check if an article with the same title already exists
            existing_article = self.collection.find_one({
                "title": processed_item['title']
            })
            if existing_article:
                self.articles_skipped += 1
                logging.info(
                    f"Duplicate article found with title: "
                    f"{processed_item['title']}. Skipping insertion."
                )
                return None

            # If no duplicate, add item to the list for bulk insertion
            self.items.append(processed_item)
            self.articles_scraped += 1
            logging.info(
                f"Article added to list for insertion: "
                f"{processed_item['title']}"
            )

        except Exception as e:
            logging.error(f"Failed to process item: {e}")

        return processed_item


class AljazeerascraperPipeline(BasePipeline):
    pass


class ArabnewsscraperPipeline(BasePipeline):
    pass


class BreitbartscraperPipeline(BasePipeline):
    pass


class FoxscraperPipeline(BasePipeline):
    pass


class MeescraperPipeline(BasePipeline):
    pass


class TimescraperPipeline(BasePipeline):
    pass


class GuardianscraperPipeline(BasePipeline):
    pass
