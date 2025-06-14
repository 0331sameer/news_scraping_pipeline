import asyncio
import aiohttp
import time
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalyticsProcessor:
    def __init__(self):
        self.mongo_uri = "mongodb+srv://jamshidjunaid763:JUNAID12345@insightwirecluster.qz5cz.mongodb.net/?retryWrites=true&w=majority&appName=InsightWireCluster"
        self.db_name = "Scraped-Articles-11"
        self.collection_name = "categorizedarticles"
        self.analytics_api_url = "http://127.0.0.1:8001/analyze/cluster"
        self.delay_seconds = 1
        
    async def connect_to_db(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    async def find_empty_analytics_clusters(self):
        """Find all clusters with empty Analytics arrays"""
        try:
            # Query for documents where Analytics field is empty or doesn't exist
            query = {
                "$or": [
                    {"Analytics": {"$exists": False}},
                    {"Analytics": {"$size": 0}},
                    {"Analytics": []}
                ]
            }
            
            clusters = []
            async for cluster in self.collection.find(query):
                clusters.append({
                    "id": str(cluster["_id"]),
                    "title": cluster.get("title", "Unknown Title"),
                    "clusterTopic": cluster.get("clusterTopic", "Unknown Topic")
                })
            
            logger.info(f"Found {len(clusters)} clusters with empty Analytics")
            return clusters
            
        except Exception as e:
            logger.error(f"Error finding empty analytics clusters: {str(e)}")
            return []
    
    async def process_cluster(self, session, cluster):
        """Process a single cluster through the analytics API"""
        try:
            cluster_id = cluster["id"]
            logger.info(f"Processing cluster: {cluster_id} - {cluster['title'][:50]}...")
            
            # Make POST request to analytics API
            url = f"{self.analytics_api_url}/{cluster_id}"
            async with session.post(url, headers={"Content-Type": "application/json"}) as response:
                if response.status == 200:
                    result = await response.json()
                    possibilities_count = len(result.get("possibilities", []))
                    logger.info(f"✅ Successfully processed cluster {cluster_id} - {possibilities_count} possibilities found")
                    return {"success": True, "cluster_id": cluster_id, "possibilities": possibilities_count}
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Failed to process cluster {cluster_id}: HTTP {response.status} - {error_text}")
                    return {"success": False, "cluster_id": cluster_id, "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Error processing cluster {cluster['id']}: {str(e)}")
            return {"success": False, "cluster_id": cluster["id"], "error": str(e)}
    
    async def process_all_clusters(self):
        """Process all clusters with empty analytics"""
        try:
            # Connect to database
            await self.connect_to_db()
            
            # Find clusters with empty analytics
            clusters = await self.find_empty_analytics_clusters()
            
            if not clusters:
                logger.info("No clusters with empty Analytics found. All clusters are already processed!")
                return
            
            logger.info(f"Starting to process {len(clusters)} clusters...")
            
            # Process clusters with delay
            results = {"success": 0, "failed": 0, "total_possibilities": 0}
            
            async with aiohttp.ClientSession() as session:
                for i, cluster in enumerate(clusters, 1):
                    logger.info(f"Progress: {i}/{len(clusters)}")
                    
                    # Process the cluster
                    result = await self.process_cluster(session, cluster)
                    
                    if result["success"]:
                        results["success"] += 1
                        results["total_possibilities"] += result["possibilities"]
                    else:
                        results["failed"] += 1
                    
                    # Add delay between requests (except for the last one)
                    if i < len(clusters):
                        logger.info(f"Waiting {self.delay_seconds} second(s) before next request...")
                        await asyncio.sleep(self.delay_seconds)
            
            # Print final summary
            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETE!")
            logger.info(f"Total clusters processed: {len(clusters)}")
            logger.info(f"Successful: {results['success']}")
            logger.info(f"Failed: {results['failed']}")
            logger.info(f"Total possibilities generated: {results['total_possibilities']}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in process_all_clusters: {str(e)}")
        finally:
            if hasattr(self, 'client'):
                self.client.close()
                logger.info("Database connection closed")

async def main():
    """Main function to run the analytics processor"""
    logger.info("Starting Analytics Processor...")
    logger.info("This script will find all categorized articles with empty Analytics arrays")
    logger.info("and process them through the newsAnalytics service.")
    logger.info("")
    
    processor = AnalyticsProcessor()
    await processor.process_all_clusters()
    
    logger.info("Analytics processing completed!")

if __name__ == "__main__":
    asyncio.run(main()) 