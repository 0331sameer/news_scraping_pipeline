import subprocess
import time
import sys

def run_spider(spider_name):
    """Run a single spider and wait for it to complete"""
    print(f"\n{'='*50}")
    print(f"Starting {spider_name} spider...")
    print(f"{'='*50}")
    
    try:
        # Run the spider using scrapy crawl command
        result = subprocess.run([
            sys.executable, '-m', 'scrapy', 'crawl', spider_name
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {spider_name} completed successfully!")
        else:
            print(f"❌ {spider_name} failed with error:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {spider_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {spider_name}: {e}")
        return False

def main():
    # List of spiders to run
    spiders = [
        'aljazeeraspider',
        'arabnewsspider',
        'breitbartspider',
        'foxspider',
        'guardianspider',
        'meespider',
        'timespider'
    ]
    
    successful_spiders = []
    failed_spiders = []
    
    print("🚀 Starting sequential news scraping...")
    print(f"Total spiders to run: {len(spiders)}")
    
    for i, spider in enumerate(spiders, 1):
        print(f"\n📊 Progress: {i}/{len(spiders)}")
        
        success = run_spider(spider)
        
        if success:
            successful_spiders.append(spider)
        else:
            failed_spiders.append(spider)
        
        # Wait between spiders to be respectful
        if i < len(spiders):
            print("⏳ Waiting 30 seconds before next spider...")
            time.sleep(30)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📈 SCRAPING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful_spiders)}")
    for spider in successful_spiders:
        print(f"   - {spider}")
    
    print(f"\n❌ Failed: {len(failed_spiders)}")
    for spider in failed_spiders:
        print(f"   - {spider}")
    
    print(f"\n🎯 Success Rate: {len(successful_spiders)}/{len(spiders)} ({len(successful_spiders)/len(spiders)*100:.1f}%)")

if __name__ == "__main__":
    main()