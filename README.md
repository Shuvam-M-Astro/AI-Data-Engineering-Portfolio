# AI Data Engineering Portfolio

A comprehensive collection of machine learning and data engineering projects organized by use cases and domains.

## 📁 Project Structure

### 🎯 Classification
- **Image Classification**
  - [CIFAR-10](./Classification/Image_Classification/CIFAR-10/) - CNN-based image classification
  - [MNIST](./Classification/Image_Classification/MNIST/) - Handwritten digit recognition with AB testing
  - [AIR](./Classification/Image_Classification/AIR/) - Advanced image recognition

- **Text Classification**
  - [IMDB](./Classification/Text_Classification/IMDB/) - Sentiment analysis on movie reviews
  - [IMDB Consolidated](./Classification/Text_Classification/IMDB_Consolidated/) - Advanced sentiment analysis with RPA

- **Tabular Classification**
  - [IRIS](./Classification/Tabular_Classification/IRIS/) - Flower species classification with pruning optimization
  - [Titanic](./Classification/Tabular_Classification/Titanic/) - Survival prediction

### 📈 Regression
- **Time Series Forecasting**
  - [Stock Price Prediction](./Regression/Time_Series_Forecasting/Stock_Price_Prediction/) - LSTM-based financial forecasting
  - [Weather Forecasting](./Regression/Time_Series_Forecasting/Weather_Forecasting/) - Meteorological data prediction
  - [Energy Consumption](./Regression/Time_Series_Forecasting/Energy_Consumption/) - Power usage forecasting

- **Linear/Non-linear Regression**
  - [Housing Price Prediction](./Regression/Linear_Regression/Housing_Price_Prediction/) - Real estate price modeling
  - [Demand Forecasting](./Regression/Linear_Regression/Demand_Forecasting/) - Sales prediction models

### 🔍 Clustering & Unsupervised Learning
- **Customer Segmentation**
  - [K-means Clustering](./Clustering/Customer_Segmentation/) - Market analysis and customer grouping
  - [Hierarchical Clustering](./Clustering/Hierarchical_Clustering/) - Dendrogram-based analysis
  - [DBSCAN](./Clustering/DBSCAN/) - Density-based clustering

- **Dimensionality Reduction**
  - [PCA Analysis](./Clustering/Dimensionality_Reduction/PCA/) - Principal component analysis
  - [t-SNE Visualization](./Clustering/Dimensionality_Reduction/t-SNE/) - High-dimensional data visualization
  - [UMAP](./Clustering/Dimensionality_Reduction/UMAP/) - Uniform manifold approximation

- **Anomaly Detection**
  - [Isolation Forest](./Clustering/Anomaly_Detection/Isolation_Forest/) - Outlier detection
  - [One-Class SVM](./Clustering/Anomaly_Detection/One_Class_SVM/) - Novelty detection

### 🎮 Reinforcement Learning
- **Q-Learning**
  - [Grid World](./Reinforcement_Learning/Q_Learning/Grid_World/) - Basic Q-learning implementation
  - [CartPole](./Reinforcement_Learning/Q_Learning/CartPole/) - OpenAI Gym environment

- **Deep Q-Network (DQN)**
  - [Atari Games](./Reinforcement_Learning/DQN/Atari_Games/) - Game playing with DQN
  - [Custom Environment](./Reinforcement_Learning/DQN/Custom_Environment/) - Custom RL environment

- **Policy Gradient Methods**
  - [REINFORCE](./Reinforcement_Learning/Policy_Gradient/REINFORCE/) - Policy gradient algorithm
  - [Actor-Critic](./Reinforcement_Learning/Policy_Gradient/Actor_Critic/) - Actor-critic implementation

### 👁️ Computer Vision
- **Object Detection**
  - [YOLO Implementation](./Computer_Vision/Object_Detection/YOLO/) - Real-time object detection
  - [Faster R-CNN](./Computer_Vision/Object_Detection/Faster_RCNN/) - Region-based detection

- **Image Segmentation**
  - [U-Net](./Computer_Vision/Image_Segmentation/U_Net/) - Medical image segmentation
  - [Mask R-CNN](./Computer_Vision/Image_Segmentation/Mask_RCNN/) - Instance segmentation

- **Face Recognition**
  - [Face Detection](./Computer_Vision/Face_Recognition/Face_Detection/) - Face detection system
  - [Face Recognition](./Computer_Vision/Face_Recognition/Face_Recognition/) - Identity verification

- **Image Generation**
  - [GANs](./Computer_Vision/Image_Generation/GANs/) - Generative adversarial networks
  - [VAEs](./Computer_Vision/Image_Generation/VAEs/) - Variational autoencoders
  - [Diffusion Models](./Computer_Vision/Image_Generation/Diffusion_Models/) - Modern image generation

### 🤖 Advanced NLP
- **Named Entity Recognition (NER)**
  - [NER System](./NLP/Named_Entity_Recognition/NER_System/) - Entity extraction pipeline
  - [Custom NER](./NLP/Named_Entity_Recognition/Custom_NER/) - Domain-specific entity recognition

- **Text Summarization**
  - [Extractive Summarization](./NLP/Text_Summarization/Extractive/) - Key sentence extraction
  - [Abstractive Summarization](./NLP/Text_Summarization/Abstractive/) - Neural summarization

- **Machine Translation**
  - [Seq2Seq Translation](./NLP/Machine_Translation/Seq2Seq/) - Sequence-to-sequence translation
  - [Transformer Translation](./NLP/Machine_Translation/Transformer/) - Transformer-based translation

- **Text-to-Speech**
  - [TTS System](./NLP/Text_to_Speech/TTS_System/) - Speech synthesis
  - [Voice Cloning](./NLP/Text_to_Speech/Voice_Cloning/) - Voice replication

### 🕸️ Graph Neural Networks
- **Node Classification**
  - [Social Network Analysis](./Graph_Neural_Networks/Node_Classification/Social_Network/) - Community detection
  - [Citation Networks](./Graph_Neural_Networks/Node_Classification/Citation_Networks/) - Academic paper classification

- **Link Prediction**
  - [Recommendation Systems](./Graph_Neural_Networks/Link_Prediction/Recommendation/) - Graph-based recommendations
  - [Knowledge Graphs](./Graph_Neural_Networks/Link_Prediction/Knowledge_Graphs/) - Knowledge graph completion

- **Graph Embeddings**
  - [Node2Vec](./Graph_Neural_Networks/Graph_Embeddings/Node2Vec/) - Node embedding techniques
  - [GraphSAGE](./Graph_Neural_Networks/Graph_Embeddings/GraphSAGE/) - Inductive graph learning

### ⏰ Time Series Analysis
- **Seasonal Decomposition**
  - [STL Decomposition](./Time_Series_Analysis/Seasonal_Decomposition/STL/) - Seasonal trend decomposition
  - [X-13ARIMA-SEATS](./Time_Series_Analysis/Seasonal_Decomposition/X13ARIMA/) - Advanced seasonal adjustment

- **Change Point Detection**
  - [Bayesian Change Points](./Time_Series_Analysis/Change_Point_Detection/Bayesian/) - Bayesian change detection
  - [Online Change Detection](./Time_Series_Analysis/Change_Point_Detection/Online/) - Real-time change detection

- **Multivariate Time Series**
  - [VAR Models](./Time_Series_Analysis/Multivariate/VAR/) - Vector autoregression
  - [VECM Models](./Time_Series_Analysis/Multivariate/VECM/) - Vector error correction

### 🎯 Recommendation Systems
- **Collaborative Filtering**
  - [Matrix Factorization](./Recommendation_Systems/Collaborative_Filtering/Matrix_Factorization/) - User-item factorization
  - [Neural CF](./Recommendation_Systems/Collaborative_Filtering/Neural_CF/) - Neural collaborative filtering

- **Content-Based Filtering**
  - [Feature-Based Recommendations](./Recommendation_Systems/Content_Based/Feature_Based/) - Content similarity
  - [Text-Based Recommendations](./Recommendation_Systems/Content_Based/Text_Based/) - Text similarity

- **Hybrid Systems**
  - [Hybrid Recommendations](./Recommendation_Systems/Hybrid/Hybrid_System/) - Combined approaches
  - [Real-time Recommendations](./Recommendation_Systems/Hybrid/Real_Time/) - Online learning systems

### 🚀 Big Data & Distributed Computing
- **Apache Spark**
  - [Large-scale Processing](./Big_Data/Apache_Spark/Large_Scale_Processing/) - Spark ML pipelines
  - [Streaming Analytics](./Big_Data/Apache_Spark/Streaming_Analytics/) - Real-time data processing

- **Dask**
  - [Parallel Computing](./Big_Data/Dask/Parallel_Computing/) - Distributed data processing
  - [ML with Dask](./Big_Data/Dask/ML_with_Dask/) - Distributed machine learning

- **Ray**
  - [Distributed ML](./Big_Data/Ray/Distributed_ML/) - Ray-based ML training
  - [Hyperparameter Tuning](./Big_Data/Ray/Hyperparameter_Tuning/) - Distributed hyperparameter optimization

### 🤖 AutoML & Neural Architecture Search
- **Auto-sklearn**
  - [Automated ML](./AutoML/Auto_sklearn/Automated_ML/) - Automated pipeline optimization
  - [Feature Engineering](./AutoML/Auto_sklearn/Feature_Engineering/) - Automated feature selection

- **Neural Architecture Search (NAS)**
  - [NAS Implementation](./AutoML/NAS/NAS_Implementation/) - Automated model design
  - [Efficient NAS](./AutoML/NAS/Efficient_NAS/) - Efficient architecture search

- **Hyperparameter Optimization**
  - [Bayesian Optimization](./AutoML/Hyperparameter_Optimization/Bayesian/) - Bayesian hyperparameter tuning
  - [Optuna Integration](./AutoML/Hyperparameter_Optimization/Optuna/) - Advanced hyperparameter optimization

### 🔍 Explainable AI (XAI)
- **SHAP Values**
  - [Model Interpretability](./Explainable_AI/SHAP/Model_Interpretability/) - SHAP-based explanations
  - [Feature Importance](./Explainable_AI/SHAP/Feature_Importance/) - Feature contribution analysis

- **LIME**
  - [Local Explanations](./Explainable_AI/LIME/Local_Explanations/) - Local interpretable explanations
  - [Text Explanations](./Explainable_AI/LIME/Text_Explanations/) - Text model explanations

- **Model Debugging**
  - [Error Analysis](./Explainable_AI/Model_Debugging/Error_Analysis/) - Model error investigation
  - [Bias Detection](./Explainable_AI/Model_Debugging/Bias_Detection/) - Algorithmic bias detection

### 📱 Edge Computing & IoT
- **Model Quantization**
  - [TensorFlow Lite](./Edge_Computing/Model_Quantization/TensorFlow_Lite/) - Mobile model optimization
  - [ONNX Models](./Edge_Computing/Model_Quantization/ONNX/) - Cross-platform model deployment

- **Federated Learning**
  - [Federated Training](./Edge_Computing/Federated_Learning/Federated_Training/) - Privacy-preserving learning
  - [Secure Aggregation](./Edge_Computing/Federated_Learning/Secure_Aggregation/) - Secure federated protocols

- **Edge ML**
  - [On-device Inference](./Edge_Computing/Edge_ML/On_device_Inference/) - Edge model deployment
  - [IoT Data Processing](./Edge_Computing/Edge_ML/IoT_Data_Processing/) - Sensor data analysis

### 🔧 Data Engineering & ETL
- **Apache Airflow**
  - [Workflow Orchestration](./Data_Engineering/Apache_Airflow/Workflow_Orchestration/) - ML pipeline orchestration
  - [Data Pipelines](./Data_Engineering/Apache_Airflow/Data_Pipelines/) - ETL pipeline automation

- **Data Quality**
  - [Great Expectations](./Data_Engineering/Data_Quality/Great_Expectations/) - Data validation framework
  - [Deequ](./Data_Engineering/Data_Quality/Deequ/) - Data quality monitoring

- **Feature Stores**
  - [Feast Integration](./Data_Engineering/Feature_Stores/Feast/) - Feature store implementation
  - [Hopsworks](./Data_Engineering/Feature_Stores/Hopsworks/) - Enterprise feature store

### 🏥 Domain-Specific Applications
- **Healthcare ML**
  - [Medical Image Analysis](./Domain_Specific/Healthcare/Medical_Image_Analysis/) - Radiology image processing
  - [Patient Risk Prediction](./Domain_Specific/Healthcare/Patient_Risk_Prediction/) - Clinical risk assessment

- **Financial ML**
  - [Algorithmic Trading](./Domain_Specific/Financial/Algorithmic_Trading/) - Trading strategy implementation
  - [Fraud Detection](./Domain_Specific/Financial/Fraud_Detection/) - Financial fraud detection

- **Retail Analytics**
  - [Demand Forecasting](./Domain_Specific/Retail/Demand_Forecasting/) - Inventory optimization
  - [Customer Analytics](./Domain_Specific/Retail/Customer_Analytics/) - Customer behavior analysis

- **Manufacturing**
  - [Predictive Maintenance](./Domain_Specific/Manufacturing/Predictive_Maintenance/) - Equipment failure prediction
  - [Quality Control](./Domain_Specific/Manufacturing/Quality_Control/) - Defect detection systems

### 🤖 NLP (Natural Language Processing)
- **Text Generation**
  - [LangChain](./NLP/Text_Generation/LangChain/) - Chatbot implementation

- **Sentiment Analysis**
  - [RPA Sentiment Analysis](./NLP/Sentiment_Analysis/RPA_Sentiment_Analysis/) - Automated sentiment analysis

- **Question Answering**
  - [RAG](./NLP/Question_Answering/RAG/) - Retrieval-Augmented Generation system

- **Speech Recognition**
  - [SpeechRecognition](./NLP/Speech_Recognition/) - Audio processing and speech recognition

### 🔧 MLOps
- **Model Serving**
  - [ModelServing](./MLOps/Model_Serving/) - FastAPI-based model deployment

- **Pipelines**
  - [MLOps Pipelines](./MLOps/Pipelines/) - End-to-end ML pipeline implementation

- **AB Testing**
  - [MNIST AB Testing](./MLOps/AB_Testing/MNIST_AB_Testing/) - A/B testing framework for ML models

### ⚡ Optimization
- **Bayesian Modelling**
  - [Bayesian Modelling](./Optimization/Bayesian_Modelling/) - Bayesian analysis for churn prediction and CLV

- **Pruning**
  - [Model Pruning](./Optimization/Pruning/) - Neural network pruning techniques

- **Benchmarking**
  - [Performance Benchmarking](./Optimization/Benchmarking/) - Model performance evaluation

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements-common.txt
```

### Running Projects
Each project directory contains its own `requirements.txt` and instructions. Navigate to any project folder and follow the README.

## 📊 Project Highlights

### Image Classification
- **CIFAR-10**: CNN implementation with data augmentation
- **MNIST**: Comprehensive AB testing framework
- **AIR**: Advanced image recognition techniques

### Text Analysis
- **IMDB**: Sentiment analysis with preprocessing
- **RPA**: Automated sentiment analysis pipeline
- **LangChain**: Modern chatbot implementation

### MLOps
- **Model Serving**: Production-ready API with authentication
- **Pipelines**: Scalable ML pipeline architecture
- **AB Testing**: Statistical framework for model comparison

### Optimization
- **Bayesian Analysis**: Uncertainty quantification for business problems
- **Pruning**: Model compression techniques
- **Benchmarking**: Performance evaluation frameworks

## 🛠️ Technologies Used

- **Machine Learning**: Scikit-learn, TensorFlow, PyTorch
- **Deep Learning**: CNN, RNN, Transformers
- **NLP**: NLTK, spaCy, LangChain
- **MLOps**: FastAPI, Docker, MLflow
- **Optimization**: Bayesian inference, model pruning
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📈 Key Features

- **Modular Design**: Each project is self-contained
- **Production Ready**: Includes deployment and serving capabilities
- **Best Practices**: Follows ML engineering standards
- **Documentation**: Comprehensive READMEs and comments
- **Testing**: Unit tests and validation frameworks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your project to the appropriate category
4. Update the main README
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or contributions, please open an issue or submit a pull request.

---

**Note**: This portfolio demonstrates various aspects of AI/ML engineering including classification, NLP, MLOps, and optimization techniques. Each project is designed to be educational and production-ready.

