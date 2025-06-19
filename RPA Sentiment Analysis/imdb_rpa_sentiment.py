#!/usr/bin/env python3
"""
IMDB RPA Sentiment Analysis Tool

A professional web scraping and sentiment analysis tool for IMDB movie reviews.
Features include:
- Robust error handling and logging
- Configuration management
- Modular design with separate classes
- Progress tracking and reporting
- Data validation and cleaning
- Testing capabilities
- Performance monitoring

Author: ML Portfolio
Version: 2.0.0
"""

import time
import logging
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from bs4 import BeautifulSoup
from transformers import pipeline
import requests
from tqdm import tqdm
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('imdb_sentiment_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MovieData:
    """Data class for movie information."""
    title: str
    year: str
    rating: str
    link: str
    num_reviews: int = 0
    avg_sentiment: Optional[float] = None
    sentiment_std: Optional[float] = None
    positive_reviews: int = 0
    negative_reviews: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None

@dataclass
class Config:
    """Configuration class for the application."""
    max_movies: int = 10
    max_reviews_per_movie: int = 20
    headless: bool = True
    timeout: int = 10
    retry_attempts: int = 3
    delay_between_requests: float = 2.0
    output_dir: str = "output"
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_text_length: int = 512
    use_multiprocessing: bool = False
    max_workers: int = 4

class IMDBScraper:
    """Handles web scraping operations for IMDB."""
    
    def __init__(self, config: Config):
        self.config = config
        self.driver = None
        self.wait = None
        self.setup_driver()
    
    def setup_driver(self):
        """Initialize the Chrome WebDriver with proper options."""
        try:
            chrome_options = Options()
            if self.config.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, self.config.timeout)
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def get_popular_movies(self) -> List[Dict]:
        """Scrape the most popular movies from IMDB."""
        url = "https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm"
        
        try:
            logger.info(f"Scraping popular movies from: {url}")
            self.driver.get(url)
            time.sleep(self.config.delay_between_requests)
            
            # Wait for the table to load
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.lister-list")))
            
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            movies = []
            
            for row in soup.select("tbody.lister-list tr"):
                try:
                    title_col = row.find("td", class_="titleColumn")
                    if not title_col:
                        continue
                    
                    title = title_col.a.text.strip()
                    year = title_col.span.text.strip("()") if title_col.span else ""
                    link = "https://www.imdb.com" + title_col.a["href"].split("?")[0]
                    
                    rating_col = row.find("td", class_="imdbRating")
                    rating = rating_col.strong.text.strip() if rating_col and rating_col.strong else ""
                    
                    movies.append({
                        "title": title,
                        "year": year,
                        "rating": rating,
                        "link": link
                    })
                except Exception as e:
                    logger.warning(f"Error parsing movie row: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(movies)} movies")
            return movies[:self.config.max_movies]
            
        except TimeoutException:
            logger.error("Timeout waiting for page to load")
            return []
        except Exception as e:
            logger.error(f"Error scraping popular movies: {e}")
            return []
    
    def get_movie_reviews(self, movie_url: str) -> List[str]:
        """Scrape reviews for a specific movie."""
        reviews_url = movie_url.rstrip("/") + "/reviews"
        
        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f"Scraping reviews from: {reviews_url} (attempt {attempt + 1})")
                self.driver.get(reviews_url)
                time.sleep(self.config.delay_between_requests)
                
                # Wait for reviews to load
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".review-container")))
                
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                review_divs = soup.select(".review-container .text.show-more__control")
                
                reviews = []
                for div in review_divs[:self.config.max_reviews_per_movie]:
                    text = div.text.strip()
                    if len(text) > 50:  # Filter out very short reviews
                        reviews.append(text)
                
                logger.debug(f"Found {len(reviews)} reviews for movie")
                return reviews
                
            except TimeoutException:
                logger.warning(f"Timeout on attempt {attempt + 1} for {reviews_url}")
                if attempt == self.config.retry_attempts - 1:
                    logger.error(f"Failed to load reviews after {self.config.retry_attempts} attempts")
                    return []
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.config.retry_attempts - 1:
                    return []
        
        return []
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")

class SentimentAnalyzer:
    """Handles sentiment analysis operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the sentiment analysis model."""
        try:
            logger.info(f"Loading sentiment analysis model: {self.config.model_name}")
            self.pipeline = pipeline("sentiment-analysis", model=self.config.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def analyze_reviews(self, reviews: List[str]) -> Tuple[float, float, int, int]:
        """Analyze sentiment for a list of reviews."""
        if not reviews:
            return 0.0, 0.0, 0, 0
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        
        for review in reviews:
            try:
                # Truncate review to max length
                truncated_review = review[:self.config.max_text_length]
                result = self.pipeline(truncated_review)[0]
                
                sentiment_score = 1 if result["label"] == "POSITIVE" else 0
                sentiments.append(sentiment_score)
                
                if sentiment_score == 1:
                    positive_count += 1
                else:
                    negative_count += 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing review: {e}")
                continue
        
        if not sentiments:
            return 0.0, 0.0, 0, 0
        
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        return avg_sentiment, sentiment_std, positive_count, negative_count

class DataProcessor:
    """Handles data processing and output operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def process_movie(self, movie: Dict, scraper: IMDBScraper, analyzer: SentimentAnalyzer) -> MovieData:
        """Process a single movie and return MovieData object."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing movie: {movie['title']}")
            
            # Get reviews
            reviews = scraper.get_movie_reviews(movie['link'])
            
            # Analyze sentiment
            avg_sentiment, sentiment_std, positive_count, negative_count = analyzer.analyze_reviews(reviews)
            
            processing_time = time.time() - start_time
            
            return MovieData(
                title=movie['title'],
                year=movie['year'],
                rating=movie['rating'],
                link=movie['link'],
                num_reviews=len(reviews),
                avg_sentiment=avg_sentiment,
                sentiment_std=sentiment_std,
                positive_reviews=positive_count,
                negative_reviews=negative_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing movie {movie['title']}: {e}")
            return MovieData(
                title=movie['title'],
                year=movie['year'],
                rating=movie['rating'],
                link=movie['link'],
                error_message=str(e)
            )
    
    def save_results(self, results: List[MovieData], filename: str = None):
        """Save results to CSV and JSON files."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imdb_sentiment_analysis_{timestamp}"
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(result) for result in results])
        
        # Save CSV
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to CSV: {csv_path}")
        
        # Save JSON with metadata
        json_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_movies": len(results),
                "config": asdict(self.config)
            },
            "results": [asdict(result) for result in results]
        }
        
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Results saved to JSON: {json_path}")
        
        return csv_path, json_path
    
    def generate_report(self, results: List[MovieData]) -> str:
        """Generate a summary report."""
        successful_results = [r for r in results if r.error_message is None]
        failed_results = [r for r in results if r.error_message is not None]
        
        report = f"""
IMDB Sentiment Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Total movies processed: {len(results)}
- Successful: {len(successful_results)}
- Failed: {len(failed_results)}

Top Movies by Sentiment Score:
"""
        
        if successful_results:
            sorted_results = sorted(successful_results, key=lambda x: x.avg_sentiment or 0, reverse=True)
            for i, result in enumerate(sorted_results[:5], 1):
                report += f"{i}. {result.title} ({result.year}) - Sentiment: {result.avg_sentiment:.3f}\n"
        
        if failed_results:
            report += "\nFailed Movies:\n"
            for result in failed_results:
                report += f"- {result.title}: {result.error_message}\n"
        
        return report

class IMDBRPASentimentAnalysis:
    """Main class for IMDB RPA Sentiment Analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scraper = None
        self.analyzer = None
        self.processor = DataProcessor(config)
    
    def run(self) -> List[MovieData]:
        """Main execution method."""
        logger.info("Starting IMDB RPA Sentiment Analysis")
        start_time = time.time()
        
        try:
            # Initialize components
            self.scraper = IMDBScraper(self.config)
            self.analyzer = SentimentAnalyzer(self.config)
            
            # Get popular movies
            movies = self.scraper.get_popular_movies()
            if not movies:
                logger.error("No movies found")
                return []
            
            # Process movies
            results = []
            if self.config.use_multiprocessing:
                results = self._process_movies_parallel(movies)
            else:
                results = self._process_movies_sequential(movies)
            
            # Save results
            csv_path, json_path = self.processor.save_results(results)
            
            # Generate and save report
            report = self.processor.generate_report(results)
            report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Report saved to: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise
        finally:
            if self.scraper:
                self.scraper.close()
    
    def _process_movies_sequential(self, movies: List[Dict]) -> List[MovieData]:
        """Process movies sequentially."""
        results = []
        for movie in tqdm(movies, desc="Processing movies"):
            result = self.processor.process_movie(movie, self.scraper, self.analyzer)
            results.append(result)
        return results
    
    def _process_movies_parallel(self, movies: List[Dict]) -> List[MovieData]:
        """Process movies in parallel (experimental)."""
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_movie = {
                executor.submit(self.processor.process_movie, movie, self.scraper, self.analyzer): movie
                for movie in movies
            }
            
            for future in tqdm(as_completed(future_to_movie), total=len(movies), desc="Processing movies"):
                result = future.result()
                results.append(result)
        
        return results

def load_config(config_path: str = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
    return Config()

def create_sample_config():
    """Create a sample configuration file."""
    config = Config()
    config_dict = asdict(config)
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    logger.info("Sample configuration file created: config.yaml")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IMDB RPA Sentiment Analysis Tool")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--max-movies", "-m", type=int, help="Maximum number of movies to process")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--create-config", action="store_true", help="Create sample configuration file")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only 2 movies)")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.max_movies:
        config.max_movies = args.max_movies
    if args.headless:
        config.headless = True
    if args.test:
        config.max_movies = 2
        config.max_reviews_per_movie = 5
        logger.info("Running in test mode")
    
    try:
        # Run analysis
        analyzer = IMDBRPASentimentAnalysis(config)
        results = analyzer.run()
        
        # Print summary
        successful = [r for r in results if r.error_message is None]
        print(f"\nAnalysis completed successfully!")
        print(f"Processed {len(results)} movies ({len(successful)} successful)")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 