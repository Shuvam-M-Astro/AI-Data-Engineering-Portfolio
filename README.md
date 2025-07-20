# Simple ML Portfolio

A collection of machine learning projects implementing various algorithms and techniques on popular datasets. This portfolio has been optimized to reduce redundancy and improve maintainability.

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/[username]/Simple-ML-Portfolio.git
cd Simple-ML-Portfolio
```

2. **Run the setup script**
```bash
python setup.py
```

3. **Activate the virtual environment**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Navigate to any project and run it**
```bash
cd MNIST
python classification.py
```

## 📁 Project Structure

### 🤖 Core ML Projects
- **MNIST**: Handwritten digit recognition with A/B testing framework
- **CIFAR-10**: Image classification with CNN architectures
- **IRIS**: Classic flower classification dataset
- **Titanic**: Survival prediction using passenger data

### 📊 Advanced ML
- **AIR**: Air quality prediction
- **Bayesian Modelling**: Bayesian approaches for CLV and churn prediction

### 🧠 AI & NLP
- **IMDB_Consolidated**: Sentiment analysis (Traditional ML + RPA scraping)
- **LangChain**: Chatbot implementation with tools and memory
- **RAG**: Retrieval Augmented Generation system for document-based QA

### 🔊 Audio Processing
- **SpeechRecognition**: Speech-to-text using Whisper model

### ⚙️ MLOps & Deployment
- **MLOps**: End-to-end ML pipeline with MLflow and DVC
- **ModelServing**: FastAPI-based model serving with authentication

## 🛠️ Project Optimizations

### ✅ Completed Improvements
- **Removed empty directories**: `Churn_Bayesian/` and `CLV_Bayesian/`
- **Consolidated dependencies**: Created `requirements-common.txt` with shared packages
- **Merged IMDB projects**: Combined traditional ML and RPA approaches
- **Created shared utilities**: `shared_utils/` for common classification tasks
- **Simplified requirements**: Each project now only includes specific dependencies

### 📦 Dependency Management
- **Common dependencies**: Core ML libraries (numpy, pandas, scikit-learn, etc.)
- **Project-specific**: Only unique dependencies per project
- **Version flexibility**: Using `>=` for better compatibility

## 🧩 Shared Components

### `shared_utils/classification_utils.py`
Common utilities for all classification projects:
- Data preprocessing
- Model evaluation
- Visualization tools
- Results saving

### `requirements-common.txt`
Shared dependencies across projects:
- Core ML libraries
- Visualization tools
- Web frameworks
- Development tools

## 🎯 Technologies Used

- **Core ML**: scikit-learn, TensorFlow, PyTorch
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Modern AI**: LangChain, Transformers, Whisper
- **MLOps**: MLflow, DVC, FastAPI
- **Web Scraping**: Selenium, BeautifulSoup

## 📈 Benefits of Optimization

1. **Reduced Redundancy**: Eliminated duplicate dependencies and empty directories
2. **Easier Maintenance**: Centralized common utilities and dependencies
3. **Faster Setup**: Single setup script installs all dependencies
4. **Better Organization**: Clear separation between common and project-specific code
5. **Improved Collaboration**: Shared utilities reduce code duplication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the setup script
5. Submit a pull request

## 📝 Notes

- All projects now use the shared utilities where applicable
- Dependencies are managed centrally with project-specific additions
- The setup script handles virtual environment creation and dependency installation
- Empty directories have been removed to reduce clutter

