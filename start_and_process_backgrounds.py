#!/usr/bin/env python3
"""
Complete Background Processing Workflow
=======================================

This script provides a complete workflow for background processing:
1. Optionally starts the Context-Background service
2. Waits for service to be ready
3. Triggers background processing
4. Shows results

Usage:
    python start_and_process_backgrounds.py [--start-service]

Options:
    --start-service    Start the Context-Background service first
"""

import requests
import json
import sys
import time
import subprocess
import argparse
import os
from datetime import datetime

def check_service_health():
    """Check if the Context-Background service is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_service():
    """Start the Context-Background service."""
    print("🚀 Starting Context-Background service...")
    
    # Check if we're in the right directory
    if not os.path.exists("Context-Background"):
        print("❌ Context-Background directory not found!")
        print("💡 Make sure you're running this from the news_scraping_pipeline directory")
        return None
    
    try:
        # Start the service in background
        process = subprocess.Popen([
            "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"
        ], cwd="Context-Background", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("⏳ Waiting for service to start...")
        
        # Wait up to 30 seconds for service to be ready
        for i in range(30):
            time.sleep(1)
            if check_service_health():
                print("✅ Service started successfully!")
                return process
            print(f"   Waiting... ({i+1}/30)")
        
        print("❌ Service failed to start within 30 seconds")
        process.terminate()
        return None
        
    except FileNotFoundError:
        print("❌ uvicorn not found. Make sure virtual environment is activated.")
        return None
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        return None

def wait_for_service():
    """Wait for the service to be ready."""
    print("⏳ Waiting for Context-Background service to be ready...")
    
    for i in range(10):
        if check_service_health():
            print("✅ Service is ready!")
            return True
        time.sleep(1)
        print(f"   Checking... ({i+1}/10)")
    
    print("❌ Service is not responding")
    return False

def process_backgrounds():
    """Trigger background processing for categories."""
    try:
        print("🚀 Starting background processing...")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
        # Make the POST request
        response = requests.post(
            "http://localhost:8000/process-backgrounds",
            timeout=60  # 60 second timeout for processing
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
        print("⏱️ Request timed out. Processing might still be running.")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error. Service might not be running.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON response from server")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Background Processing Workflow')
    parser.add_argument('--start-service', action='store_true', 
                       help='Start the Context-Background service first')
    args = parser.parse_args()
    
    print("🎯 Complete Background Processing Workflow")
    print("=" * 50)
    
    service_process = None
    
    try:
        if args.start_service:
            service_process = start_service()
            if not service_process:
                print("💥 Failed to start service!")
                sys.exit(1)
        else:
            # Check if service is already running
            print("🔍 Checking if Context-Background service is running...")
            if not wait_for_service():
                print("💡 Service is not running. Use --start-service to start it automatically.")
                print("   Or start it manually: cd Context-Background && uvicorn main:app --reload")
                sys.exit(1)
        
        # Process backgrounds
        success = process_backgrounds()
        
        print("-" * 50)
        if success:
            print("🎉 Background processing completed successfully!")
            print("💡 Check your MongoDB 'categorizedarticles' collection for updated Background fields.")
        else:
            print("💥 Background processing failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        # Clean up service if we started it
        if service_process:
            print("🛑 Stopping Context-Background service...")
            service_process.terminate()
            service_process.wait()
            print("✅ Service stopped")

if __name__ == "__main__":
    main() 