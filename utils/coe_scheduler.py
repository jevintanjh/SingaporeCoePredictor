import time
import threading
import schedule
from datetime import datetime, timedelta
import requests
import trafilatura
import pandas as pd
import re
from .data_updater import COEDataUpdater
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COEScheduler:
    """
    Automated COE data update scheduler that follows official bidding calendar
    """
    
    def __init__(self):
        self.updater = COEDataUpdater()
        self.bidding_dates = []
        self.is_running = False
        self.scheduler_thread = None
        
        # 2025 COE bidding dates (backup if scraping fails)
        self.fallback_2025_dates = [
            "2025-01-02", "2025-01-15", "2025-02-05", "2025-02-19",
            "2025-03-05", "2025-03-19", "2025-04-02", "2025-04-16",
            "2025-05-07", "2025-05-21", "2025-06-04", "2025-06-18",
            "2025-07-02", "2025-07-16", "2025-08-06", "2025-08-20",
            "2025-09-03", "2025-09-17", "2025-10-01", "2025-10-15",
            "2025-11-05", "2025-11-19", "2025-12-03", "2025-12-17"
        ]
    
    def scrape_bidding_schedule(self):
        """
        Scrape COE bidding dates from official sources
        """
        urls_to_try = [
            "https://www.dbs.com.sg/personal/marketplaces/content/article/articles-car-coe-open-bidding-dates-for-2025",
            "https://onemotoring.lta.gov.sg/content/onemotoring/home/buying/coe.html",
            "https://www.lta.gov.sg/content/ltagov/en/roads-and-motoring/owning-a-vehicle/costs-of-owning-a-vehicle/certificate-of-entitlement-coe.html"
        ]
        
        logger.info("Scraping COE bidding schedule from official sources...")
        
        for url in urls_to_try:
            try:
                logger.info(f"Attempting to scrape: {url}")
                downloaded = trafilatura.fetch_url(url)
                
                if downloaded:
                    text_content = trafilatura.extract(downloaded)
                    if text_content:
                        dates = self.extract_dates_from_content(text_content)
                        if dates:
                            logger.info(f"Successfully extracted {len(dates)} bidding dates")
                            return dates
                            
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                continue
        
        logger.warning("Could not scrape current bidding dates, using fallback schedule")
        return self.get_fallback_dates()
    
    def extract_dates_from_content(self, content):
        """
        Extract bidding dates from scraped content using pattern matching
        """
        dates = []
        current_year = datetime.now().year
        
        # Pattern to match various date formats
        date_patterns = [
            r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b',
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'
        ]
        
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
            'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
            'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
            'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()
                    
                    if len(groups) == 3:
                        if pattern.startswith(r'\b(\d{4})'):  # YYYY-MM-DD format
                            year, month, day = groups
                        elif pattern.endswith(r'(\d{4})\b'):  # DD MMM YYYY format
                            day, month_str, year = groups
                            month = month_map.get(month_str.lower())
                            if not month:
                                continue
                        else:  # MM/DD/YYYY format
                            month, day, year = groups
                        
                        # Only include current and future years
                        if int(year) >= current_year:
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            try:
                                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                                if date_obj >= datetime.now() - timedelta(days=30):  # Include recent past
                                    dates.append(date_str)
                            except ValueError:
                                continue
                                
                except Exception as e:
                    logger.debug(f"Error parsing date match: {str(e)}")
                    continue
        
        # Remove duplicates and sort
        dates = sorted(list(set(dates)))
        
        # Filter to reasonable COE dates (typically 1st and 15th or nearby)
        filtered_dates = []
        for date_str in dates:
            day = int(date_str.split('-')[2])
            # COE bidding typically happens between 1st-7th and 15th-21st of month
            if (1 <= day <= 7) or (15 <= day <= 21):
                filtered_dates.append(date_str)
        
        return filtered_dates[:50]  # Limit to reasonable number
    
    def get_fallback_dates(self):
        """
        Generate fallback dates based on typical COE schedule
        """
        current_date = datetime.now()
        current_year = current_date.year
        
        # Use 2025 fallback if current year is 2025
        if current_year == 2025:
            return [date for date in self.fallback_2025_dates if datetime.strptime(date, "%Y-%m-%d") >= current_date]
        
        # Generate dates for current and next year
        dates = []
        for year in [current_year, current_year + 1]:
            for month in range(1, 13):
                # First exercise around 1st-7th
                for day in [2, 3, 4, 5]:
                    try:
                        date_obj = datetime(year, month, day)
                        if date_obj >= current_date:
                            dates.append(date_obj.strftime("%Y-%m-%d"))
                        break
                    except ValueError:
                        continue
                
                # Second exercise around 15th-21st
                for day in [15, 16, 17, 18]:
                    try:
                        date_obj = datetime(year, month, day)
                        if date_obj >= current_date:
                            dates.append(date_obj.strftime("%Y-%m-%d"))
                        break
                    except ValueError:
                        continue
        
        return dates
    
    def schedule_updates(self):
        """
        Schedule automatic updates based on COE bidding dates
        """
        self.bidding_dates = self.scrape_bidding_schedule()
        
        if not self.bidding_dates:
            logger.error("No bidding dates found, cannot schedule updates")
            return False
        
        logger.info(f"Scheduling updates for {len(self.bidding_dates)} bidding dates")
        
        # Clear existing scheduled jobs
        schedule.clear()
        
        # Schedule updates 2-3 hours after each bidding date (results typically available by then)
        for date_str in self.bidding_dates:
            try:
                bidding_date = datetime.strptime(date_str, "%Y-%m-%d")
                update_time = bidding_date + timedelta(hours=3)  # Results usually available 2-3 hours after bidding
                
                # Only schedule future updates
                if update_time > datetime.now():
                    schedule.every().day.at(update_time.strftime("%H:%M")).do(
                        self.run_scheduled_update, date_str
                    ).tag(f"coe_update_{date_str}")
                    
                    logger.info(f"Scheduled update for {date_str} at {update_time.strftime('%Y-%m-%d %H:%M')}")
                    
            except Exception as e:
                logger.error(f"Error scheduling update for {date_str}: {str(e)}")
        
        # Also schedule daily checks for new bidding dates (for years after 2025)
        schedule.every().day.at("06:00").do(self.refresh_schedule).tag("schedule_refresh")
        
        return True
    
    def run_scheduled_update(self, bidding_date):
        """
        Execute scheduled COE data update
        """
        logger.info(f"Running scheduled COE data update for bidding date: {bidding_date}")
        
        try:
            success = self.updater.update_coe_database()
            if success:
                logger.info(f"Successfully updated COE database after {bidding_date} bidding")
            else:
                logger.warning(f"COE database update failed for {bidding_date}")
                
        except Exception as e:
            logger.error(f"Error during scheduled update for {bidding_date}: {str(e)}")
    
    def refresh_schedule(self):
        """
        Refresh the bidding schedule by scraping new dates
        """
        logger.info("Refreshing COE bidding schedule...")
        
        try:
            new_dates = self.scrape_bidding_schedule()
            if new_dates and new_dates != self.bidding_dates:
                logger.info("Found updated bidding schedule, rescheduling updates")
                self.bidding_dates = new_dates
                self.schedule_updates()
            else:
                logger.info("No changes to bidding schedule")
                
        except Exception as e:
            logger.error(f"Error refreshing schedule: {str(e)}")
    
    def start_scheduler(self):
        """
        Start the background scheduler
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("Starting COE automatic update scheduler...")
        
        # Set up initial schedule
        if not self.schedule_updates():
            logger.error("Failed to set up initial schedule")
            return False
        
        # Start scheduler thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("COE scheduler started successfully")
        return True
    
    def stop_scheduler(self):
        """
        Stop the background scheduler
        """
        logger.info("Stopping COE scheduler...")
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("COE scheduler stopped")
    
    def _scheduler_loop(self):
        """
        Background thread loop for running scheduled tasks
        """
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def get_next_update_time(self):
        """
        Get the next scheduled update time
        """
        jobs = schedule.get_jobs()
        if not jobs:
            return None
        
        next_times = []
        for job in jobs:
            if hasattr(job, 'next_run'):
                next_times.append(job.next_run)
        
        return min(next_times) if next_times else None
    
    def get_status(self):
        """
        Get scheduler status information
        """
        return {
            'is_running': self.is_running,
            'bidding_dates_count': len(self.bidding_dates),
            'next_bidding_dates': self.bidding_dates[:5],  # Next 5 dates
            'scheduled_jobs_count': len(schedule.get_jobs()),
            'next_update_time': self.get_next_update_time()
        }


# Global scheduler instance
coe_scheduler = COEScheduler()

def start_coe_scheduler():
    """
    Start the global COE scheduler
    """
    return coe_scheduler.start_scheduler()

def stop_coe_scheduler():
    """
    Stop the global COE scheduler
    """
    coe_scheduler.stop_scheduler()

def get_scheduler_status():
    """
    Get the current scheduler status
    """
    return coe_scheduler.get_status()