#!/usr/bin/env python3
"""
Background Processing Runner
============================

This script automatically triggers the background processing for news categories.
It makes a POST request to the Context-Background service to process category backgrounds.

Usage:
    python run_background_processing.py

Requirements:
    - Context-Background service must be running on http://localhost:8000
    - Virtual environment should be activated
"""

import requests
import json
import sys
from datetime import datetime

def check_service_health():
    """Check if the Context-Background service is running."""
    try:
        response = requests.get("https://web-production-c91f3.up.railway.app/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def process_backgrounds():
    """Trigger background processing for categories."""
    try:
        print("🚀 Starting background processing...")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
        # Make the POST request
        response = requests.post(
            "https://web-production-c91f3.up.railway.app/process-backgrounds",
            timeout=1200  # 20 minute timeout for processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Background processing completed successfully!")
            print(f"📊 Results:")
            print(f"   • Processed: {result.get('processed_count', 0)} categories")
            print(f"   • Updated: {result.get('updated_count', 0)} categories")
            print(f"   • Set to 'Not': {result.get('set_to_not_count', 0)} categories")
            print(f"💬 Message: {result.get('message', 'No message')}")
            
            return True
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out. Processing might still be running in background.")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error. Is the Context-Background service running?")
        print("💡 Check Railway deployment at: https://web-production-c91f3.up.railway.app")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON response from server")
        return False

def main():
    """Main function to run background processing."""
    print("🎯 Background Processing Runner")
    print("=" * 50)
    
    # Check if service is running
    print("🔍 Checking if Context-Background service is running...")
    if not check_service_health():
        print("⚠️  Service health check failed. Attempting to process anyway...")
    else:
        print("✅ Service is healthy!")
    
    # Process backgrounds
    success = process_backgrounds()
    
    print("-" * 50)
    if success:
        print("🎉 Background processing completed successfully!")
        print("💡 You can now check your MongoDB 'categorizedarticles' collection")
        print("   to see the updated Background fields.")
    else:
        print("💥 Background processing failed!")
        print("🔧 Troubleshooting:")
        print("   1. Check Railway deployment status:")
        print("      https://web-production-c91f3.up.railway.app/health")
        print("   2. Check if MongoDB is accessible")
        print("   3. Verify Railway environment variables are set")
        
        sys.exit(1)

if __name__ == "__main__":
    main() 