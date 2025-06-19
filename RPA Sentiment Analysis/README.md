# IMDB RPA Sentiment Analysis Tool

A professional web scraping and sentiment analysis tool for IMDB movie reviews. This tool automatically scrapes popular movies from IMDB, extracts user reviews, and performs sentiment analysis to provide insights into public opinion about movies.

## 🚀 Features

### Core Functionality
- **Automated Web Scraping**: Scrapes IMDB's most popular movies and their user reviews
- **Sentiment Analysis**: Uses state-of-the-art transformer models for accurate sentiment classification
- **Robust Error Handling**: Comprehensive error handling and retry mechanisms
- **Progress Tracking**: Real-time progress bars and detailed logging

### Professional Features
- **Modular Architecture**: Clean, maintainable code with separate classes for different responsibilities
- **Configuration Management**: YAML-based configuration with command-line overrides
- **Comprehensive Logging**: Detailed logging with file and console output
- **Data Validation**: Input validation and data cleaning
- **Performance Monitoring**: Processing time tracking and performance metrics
- **Multiple Output Formats**: CSV, JSON, and detailed reports
- **Visualization Support**: Built-in visualization capabilities with matplotlib and Plotly

### Advanced Capabilities
- **Parallel Processing**: Optional multi-threading for faster processing
- **Retry Mechanisms**: Automatic retry on network failures
- **Rate Limiting**: Configurable delays to respect website policies
- **Headless Mode**: Browser automation without GUI for server deployment
- **Testing Suite**: Comprehensive unit and integration tests

## 📋 Requirements

- Python 3.8+
- Chrome browser (for Selenium WebDriver)
- Internet connection

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RPA-Sentiment-Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Chrome WebDriver**:
   The tool uses Selenium with Chrome. Make sure you have Chrome installed and the appropriate WebDriver version.

## 🚀 Quick Start

### Basic Usage

Run the tool with default settings:
```bash
python imdb_rpa_sentiment.py
```

### Test Mode

Run with limited data for testing:
```bash
python imdb_rpa_sentiment.py --test
```

### Custom Configuration

Create a configuration file:
```bash
python imdb_rpa_sentiment.py --create-config
```

Edit the generated `config.yaml` file and run:
```bash
python imdb_rpa_sentiment.py --config config.yaml
```

## 📖 Usage Examples

### Command Line Options

```bash
# Process only 5 movies
python imdb_rpa_sentiment.py --max-movies 5

# Run in non-headless mode (shows browser)
python imdb_rpa_sentiment.py --headless false

# Use custom configuration file
python imdb_rpa_sentiment.py --config my_config.yaml

# Test mode (process 2 movies with 5 reviews each)
python imdb_rpa_sentiment.py --test
```

### Configuration File Example

```yaml
# config.yaml
max_movies: 15
max_reviews_per_movie: 25
headless: true
timeout: 15
retry_attempts: 3
delay_between_requests: 2.5
output_dir: "my_results"
model_name: "distilbert-base-uncased-finetuned-sst-2-english"
max_text_length: 512
use_multiprocessing: false
max_workers: 4
```

## 📊 Output

The tool generates several output files:

### Data Files
- **CSV File**: `imdb_sentiment_analysis_YYYYMMDD_HHMMSS.csv`
  - Contains all movie data with sentiment scores
  - Columns: title, year, rating, link, num_reviews, avg_sentiment, etc.

- **JSON File**: `imdb_sentiment_analysis_YYYYMMDD_HHMMSS.json`
  - Structured data with metadata
  - Includes configuration and processing information

### Reports
- **Text Report**: `report_YYYYMMDD_HHMMSS.txt`
  - Summary of analysis results
  - Top movies by sentiment score
  - Failed movies and error messages

### Logs
- **Log File**: `imdb_sentiment_analysis.log`
  - Detailed execution logs
  - Error messages and debugging information

## 📈 Visualization

Generate visualizations from your results:

```bash
python visualization.py results.csv --output-dir charts
```

This creates:
- Sentiment distribution plots
- Top movies charts
- Interactive Plotly dashboards
- Performance metrics

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_imdb_sentiment.py
```

The test suite includes:
- Unit tests for all components
- Integration tests for the complete workflow
- Mock testing for external dependencies
- Configuration loading tests

## 🏗️ Architecture

### Class Structure

```
IMDBRPASentimentAnalysis (Main Controller)
├── IMDBScraper (Web Scraping)
├── SentimentAnalyzer (NLP Processing)
└── DataProcessor (Data Management)
```

### Key Classes

- **`IMDBScraper`**: Handles web scraping operations
- **`SentimentAnalyzer`**: Manages sentiment analysis pipeline
- **`DataProcessor`**: Processes and saves results
- **`Config`**: Configuration management
- **`MovieData`**: Data structure for movie information

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_movies` | 10 | Maximum number of movies to process |
| `max_reviews_per_movie` | 20 | Maximum reviews per movie |
| `headless` | true | Run browser in headless mode |
| `timeout` | 10 | WebDriver timeout in seconds |
| `retry_attempts` | 3 | Number of retry attempts |
| `delay_between_requests` | 2.0 | Delay between requests in seconds |
| `output_dir` | "output" | Output directory |
| `model_name` | "distilbert-base-uncased-finetuned-sst-2-english" | Sentiment model |
| `max_text_length` | 512 | Maximum text length for analysis |
| `use_multiprocessing` | false | Enable parallel processing |
| `max_workers` | 4 | Number of worker threads |

## 🚨 Error Handling

The tool includes comprehensive error handling:

- **Network Errors**: Automatic retry with exponential backoff
- **Parsing Errors**: Graceful handling of malformed HTML
- **Model Errors**: Fallback mechanisms for sentiment analysis
- **File System Errors**: Safe file operations with error recovery

## 📝 Logging

The tool provides detailed logging at multiple levels:

- **INFO**: General progress information
- **DEBUG**: Detailed debugging information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors that affect processing

Logs are saved to both console and file for easy debugging.

## 🔒 Ethical Considerations

- **Rate Limiting**: Built-in delays to respect website policies
- **User Agent**: Proper user agent identification
- **Robots.txt**: Respects website robots.txt policies
- **Data Usage**: Only processes publicly available data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IMDB for providing movie data
- Hugging Face for transformer models
- Selenium for web automation
- The open-source community for various dependencies

## 📞 Support

For issues and questions:
1. Check the logs for error messages
2. Review the configuration options
3. Run in test mode to isolate issues
4. Create an issue with detailed information

## 🔄 Version History

- **v2.0.0**: Professional version with comprehensive features
- **v1.0.0**: Basic functionality

---

**Note**: This tool is for educational and research purposes. Please respect IMDB's terms of service and use responsibly. 