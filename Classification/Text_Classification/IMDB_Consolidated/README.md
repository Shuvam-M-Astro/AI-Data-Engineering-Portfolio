# IMDB Consolidated Project

This directory consolidates IMDB-related projects from the original `IMDB/` and `RPA Sentiment Analysis/` directories.

## Project Structure

### 1. Traditional ML Approach (`traditional_ml/`)
- **Purpose**: Traditional machine learning classification of IMDB reviews
- **Files**: 
  - `classification.py` - ML-based sentiment analysis
  - `conv.py` - Convolutional approach for text classification
- **Method**: Uses traditional ML algorithms (SVM, Random Forest, etc.)

### 2. RPA Web Scraping Approach (`rpa_scraping/`)
- **Purpose**: Web scraping IMDB reviews and performing sentiment analysis
- **Files**:
  - `imdb_rpa_sentiment.py` - Main RPA scraping script
  - `debug_tools.py` - Debugging utilities
  - `visualization.py` - Data visualization tools
  - `test_imdb_sentiment.py` - Testing framework
- **Method**: Uses Selenium to scrape IMDB website and analyze reviews

## Usage

### For Traditional ML:
```bash
cd traditional_ml
python classification.py
```

### For RPA Scraping:
```bash
cd rpa_scraping
python imdb_rpa_sentiment.py
```

## Requirements
Install the common requirements first:
```bash
pip install -r ../requirements-common.txt
```

Then install project-specific requirements:
```bash
pip install -r requirements.txt
```

## Key Differences

1. **Data Source**: 
   - Traditional ML uses pre-downloaded dataset
   - RPA scrapes live IMDB website

2. **Approach**:
   - Traditional ML uses statistical methods
   - RPA uses web automation + NLP

3. **Use Cases**:
   - Traditional ML: Batch processing of existing data
   - RPA: Real-time sentiment analysis of current reviews 