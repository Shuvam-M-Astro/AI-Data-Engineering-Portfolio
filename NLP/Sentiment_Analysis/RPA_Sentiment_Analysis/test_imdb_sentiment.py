#!/usr/bin/env python3
"""
Test suite for IMDB RPA Sentiment Analysis Tool

This module contains comprehensive tests for all components of the IMDB sentiment analysis tool.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np

# Import the classes to test
from imdb_rpa_sentiment import (
    Config, MovieData, IMDBScraper, SentimentAnalyzer, 
    DataProcessor, IMDBRPASentimentAnalysis, load_config
)

class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        self.assertEqual(config.max_movies, 10)
        self.assertEqual(config.max_reviews_per_movie, 20)
        self.assertTrue(config.headless)
        self.assertEqual(config.timeout, 10)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = Config(max_movies=5, headless=False)
        self.assertEqual(config.max_movies, 5)
        self.assertFalse(config.headless)

class TestMovieData(unittest.TestCase):
    """Test cases for MovieData class."""
    
    def test_movie_data_creation(self):
        """Test creating MovieData object."""
        movie = MovieData(
            title="Test Movie",
            year="2023",
            rating="8.5",
            link="https://example.com",
            num_reviews=10,
            avg_sentiment=0.8
        )
        self.assertEqual(movie.title, "Test Movie")
        self.assertEqual(movie.avg_sentiment, 0.8)
    
    def test_movie_data_with_error(self):
        """Test MovieData with error message."""
        movie = MovieData(
            title="Failed Movie",
            year="2023",
            rating="",
            link="https://example.com",
            error_message="Connection timeout"
        )
        self.assertIsNotNone(movie.error_message)
        self.assertEqual(movie.error_message, "Connection timeout")

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    @patch('imdb_rpa_sentiment.pipeline')
    def test_model_loading(self, mock_pipeline):
        """Test sentiment model loading."""
        mock_pipeline.return_value = Mock()
        analyzer = SentimentAnalyzer(self.config)
        self.assertIsNotNone(analyzer.pipeline)
        mock_pipeline.assert_called_once()
    
    @patch('imdb_rpa_sentiment.pipeline')
    def test_analyze_reviews(self, mock_pipeline):
        """Test review sentiment analysis."""
        # Mock the pipeline to return positive sentiment
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value = [{"label": "POSITIVE", "score": 0.9}]
        
        analyzer = SentimentAnalyzer(self.config)
        reviews = ["This movie is amazing!", "Great film!"]
        
        avg_sentiment, sentiment_std, positive_count, negative_count = analyzer.analyze_reviews(reviews)
        
        self.assertEqual(avg_sentiment, 1.0)
        self.assertEqual(positive_count, 2)
        self.assertEqual(negative_count, 0)
    
    @patch('imdb_rpa_sentiment.pipeline')
    def test_analyze_empty_reviews(self, mock_pipeline):
        """Test analysis with empty reviews list."""
        mock_pipeline.return_value = Mock()
        analyzer = SentimentAnalyzer(self.config)
        
        avg_sentiment, sentiment_std, positive_count, negative_count = analyzer.analyze_reviews([])
        
        self.assertEqual(avg_sentiment, 0.0)
        self.assertEqual(positive_count, 0)
        self.assertEqual(negative_count, 0)

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.output_dir = "test_output"
        self.processor = DataProcessor(self.config)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
    
    def test_output_directory_creation(self):
        """Test output directory creation."""
        self.assertTrue(os.path.exists(self.config.output_dir))
    
    def test_save_results(self):
        """Test saving results to files."""
        results = [
            MovieData(
                title="Test Movie 1",
                year="2023",
                rating="8.5",
                link="https://example.com/1",
                avg_sentiment=0.8
            ),
            MovieData(
                title="Test Movie 2",
                year="2023",
                rating="7.5",
                link="https://example.com/2",
                avg_sentiment=0.6
            )
        ]
        
        csv_path, json_path = self.processor.save_results(results, "test_results")
        
        # Check if files were created
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        
        # Check CSV content
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        self.assertIn("title", df.columns)
        self.assertIn("avg_sentiment", df.columns)
        
        # Check JSON content
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.assertIn("metadata", json_data)
        self.assertIn("results", json_data)
        self.assertEqual(len(json_data["results"]), 2)
    
    def test_generate_report(self):
        """Test report generation."""
        results = [
            MovieData(
                title="Good Movie",
                year="2023",
                rating="8.5",
                link="https://example.com/1",
                avg_sentiment=0.9
            ),
            MovieData(
                title="Bad Movie",
                year="2023",
                rating="3.5",
                link="https://example.com/2",
                avg_sentiment=0.2
            ),
            MovieData(
                title="Failed Movie",
                year="2023",
                rating="",
                link="https://example.com/3",
                error_message="Connection error"
            )
        ]
        
        report = self.processor.generate_report(results)
        
        self.assertIn("Good Movie", report)
        self.assertIn("Bad Movie", report)
        self.assertIn("Failed Movie", report)
        self.assertIn("Connection error", report)
        self.assertIn("Total movies processed: 3", report)
        self.assertIn("Successful: 2", report)
        self.assertIn("Failed: 1", report)

class TestIMDBScraper(unittest.TestCase):
    """Test cases for IMDBScraper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.headless = True
    
    @patch('imdb_rpa_sentiment.webdriver.Chrome')
    def test_driver_setup(self, mock_chrome):
        """Test WebDriver setup."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        scraper = IMDBScraper(self.config)
        
        self.assertIsNotNone(scraper.driver)
        self.assertIsNotNone(scraper.wait)
        mock_chrome.assert_called_once()
    
    @patch('imdb_rpa_sentiment.webdriver.Chrome')
    def test_get_popular_movies_mock(self, mock_chrome):
        """Test getting popular movies with mocked data."""
        # Mock the driver and page source
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        # Mock HTML content
        mock_html = """
        <html>
            <tbody class="lister-list">
                <tr>
                    <td class="titleColumn">
                        <a href="/title/tt1234567/">Test Movie</a>
                        <span class="secondaryInfo">(2023)</span>
                    </td>
                    <td class="imdbRating">
                        <strong>8.5</strong>
                    </td>
                </tr>
            </tbody>
        </html>
        """
        mock_driver.page_source = mock_html
        
        scraper = IMDBScraper(self.config)
        scraper.driver = mock_driver
        scraper.wait = Mock()
        
        movies = scraper.get_popular_movies()
        
        self.assertEqual(len(movies), 1)
        self.assertEqual(movies[0]["title"], "Test Movie")
        self.assertEqual(movies[0]["year"], "2023")
        self.assertEqual(movies[0]["rating"], "8.5")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.max_movies = 2
        self.config.max_reviews_per_movie = 3
        self.config.headless = True
    
    @patch('imdb_rpa_sentiment.webdriver.Chrome')
    @patch('imdb_rpa_sentiment.pipeline')
    def test_complete_workflow_mock(self, mock_pipeline, mock_chrome):
        """Test complete workflow with mocked components."""
        # Mock WebDriver
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        # Mock HTML content for movies
        mock_movies_html = """
        <html>
            <tbody class="lister-list">
                <tr>
                    <td class="titleColumn">
                        <a href="/title/tt1234567/">Test Movie 1</a>
                        <span class="secondaryInfo">(2023)</span>
                    </td>
                    <td class="imdbRating">
                        <strong>8.5</strong>
                    </td>
                </tr>
                <tr>
                    <td class="titleColumn">
                        <a href="/title/tt7654321/">Test Movie 2</a>
                        <span class="secondaryInfo">(2023)</span>
                    </td>
                    <td class="imdbRating">
                        <strong>7.5</strong>
                    </td>
                </tr>
            </tbody>
        </html>
        """
        
        # Mock HTML content for reviews
        mock_reviews_html = """
        <html>
            <div class="review-container">
                <div class="text show-more__control">Great movie!</div>
                <div class="text show-more__control">Amazing film!</div>
                <div class="text show-more__control">Excellent!</div>
            </div>
        </html>
        """
        
        mock_driver.page_source = mock_movies_html
        
        # Mock sentiment analysis
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value = [{"label": "POSITIVE", "score": 0.9}]
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config.output_dir = temp_dir
            
            # Run the analysis
            analyzer = IMDBRPASentimentAnalysis(self.config)
            
            # Mock the scraper to return our test data
            with patch.object(analyzer, 'scraper') as mock_scraper:
                mock_scraper.get_popular_movies.return_value = [
                    {
                        "title": "Test Movie 1",
                        "year": "2023",
                        "rating": "8.5",
                        "link": "https://imdb.com/title/tt1234567/"
                    },
                    {
                        "title": "Test Movie 2",
                        "year": "2023",
                        "rating": "7.5",
                        "link": "https://imdb.com/title/tt7654321/"
                    }
                ]
                mock_scraper.get_movie_reviews.return_value = [
                    "Great movie!", "Amazing film!", "Excellent!"
                ]
                
                results = analyzer.run()
                
                # Verify results
                self.assertEqual(len(results), 2)
                self.assertIsNone(results[0].error_message)
                self.assertIsNone(results[1].error_message)

class TestConfigurationLoading(unittest.TestCase):
    """Test cases for configuration loading."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.max_movies, 10)
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        # Create temporary config file
        config_data = {
            "max_movies": 5,
            "max_reviews_per_movie": 10,
            "headless": False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            self.assertEqual(config.max_movies, 5)
            self.assertEqual(config.max_reviews_per_movie, 10)
            self.assertFalse(config.headless)
        finally:
            os.unlink(config_path)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConfig,
        TestMovieData,
        TestSentimentAnalyzer,
        TestDataProcessor,
        TestIMDBScraper,
        TestIntegration,
        TestConfigurationLoading
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 