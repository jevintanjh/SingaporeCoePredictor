#!/usr/bin/env python3
"""
Keep-alive script for Streamlit app to prevent sleeping due to inactivity
"""

import requests
import time
import threading
import schedule
from datetime import datetime

class StreamlitKeepAlive:
    def __init__(self, app_url="http://localhost:8501", ping_interval=900):  # 15 minutes
        self.app_url = app_url
        self.ping_interval = ping_interval
        self.running = False
        self.thread = None
    
    def ping_app(self):
        """Send a simple GET request to keep the app alive"""
        try:
            response = requests.get(self.app_url, timeout=10)
            status = "✅ Success" if response.status_code == 200 else f"⚠️ Status {response.status_code}"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Keep-alive ping: {status}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Keep-alive error: {str(e)}")
    
    def start(self):
        """Start the keep-alive service"""
        if self.running:
            print("Keep-alive service is already running")
            return
        
        self.running = True
        print(f"Starting keep-alive service for {self.app_url}")
        print(f"Ping interval: {self.ping_interval} seconds")
        
        # Schedule pings
        schedule.every(self.ping_interval // 60).minutes.do(self.ping_app)
        
        # Run scheduler in background thread
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.thread = threading.Thread(target=run_scheduler, daemon=True)
        self.thread.start()
        
        # Initial ping
        self.ping_app()
    
    def stop(self):
        """Stop the keep-alive service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Keep-alive service stopped")

# Global instance
keep_alive_service = StreamlitKeepAlive()

def start_keep_alive():
    """Start the keep-alive service"""
    keep_alive_service.start()

def stop_keep_alive():
    """Stop the keep-alive service"""
    keep_alive_service.stop()

if __name__ == "__main__":
    # For standalone execution
    keep_alive = StreamlitKeepAlive()
    keep_alive.start()
    
    try:
        # Keep the script running
        while True:
            time.sleep(3600)  # Sleep for 1 hour
    except KeyboardInterrupt:
        keep_alive.stop()
        print("Keep-alive service terminated")