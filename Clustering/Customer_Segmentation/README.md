# Customer Segmentation Analysis

## Overview
This project implements advanced customer segmentation using K-means clustering to identify distinct customer groups based on their behavior patterns. The system includes RFM analysis, comprehensive visualizations, and marketing strategy recommendations.

## Features

### Core Functionality
- **Data Generation**: Synthetic customer data generation for demonstration
- **RFM Analysis**: Recency, Frequency, Monetary value analysis
- **Feature Engineering**: Comprehensive feature preparation and scaling
- **Optimal Clustering**: Automatic determination of optimal cluster count using elbow method and silhouette analysis
- **Cluster Analysis**: Detailed analysis of cluster characteristics

### Advanced Analytics
- **PCA Visualization**: Dimensionality reduction for cluster visualization
- **Interactive Plots**: Plotly-based interactive visualizations
- **Customer Profiles**: Detailed customer segment profiles
- **Marketing Strategies**: Automated marketing strategy recommendations

### Visualizations
- PCA cluster visualization
- Cluster size distribution
- Feature importance heatmap
- RFM 3D scatter plot
- Elbow method and silhouette analysis plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from customer_segmentation import CustomerSegmentation

# Initialize the model
segmentation = CustomerSegmentation()

# Generate sample data
data = segmentation.generate_sample_data(n_customers=1000)

# Add RFM features
segmentation.add_rfm_features()

# Prepare features
features = ['age', 'income', 'spending_score', 'purchase_frequency',
           'avg_order_value', 'total_purchases', 'online_visits',
           'mobile_app_usage', 'customer_satisfaction', 'rfm_score']
scaled_data, feature_names = segmentation.prepare_features(features)

# Find optimal clusters
optimal_k = segmentation.find_optimal_clusters(max_clusters=8)

# Perform clustering
segmentation.perform_clustering(n_clusters=optimal_k)

# Analyze and visualize
cluster_analysis = segmentation.analyze_clusters(features)
segmentation.visualize_clusters(features)

# Create profiles and strategies
profiles = segmentation.create_customer_profiles(features)
strategies = segmentation.recommend_strategies(profiles)
```

### Run Complete Analysis
```bash
python customer_segmentation.py
```

## Output

### Cluster Analysis
The system provides detailed cluster analysis including:
- Cluster sizes and percentages
- Average feature values per cluster
- Customer profile descriptions
- Marketing strategy recommendations

### Visualizations
- Interactive PCA cluster plots
- Cluster distribution charts
- Feature importance heatmaps
- RFM analysis 3D plots

### Marketing Insights
For each cluster, the system provides:
- **Targeting**: Customer segment identification
- **Messaging**: Recommended communication approach
- **Offers**: Suggested promotional strategies
- **Channels**: Preferred marketing channels

## Features Used

### Customer Demographics
- Age
- Income
- Customer satisfaction

### Behavioral Metrics
- Spending score
- Purchase frequency
- Average order value
- Total purchases
- Online visits
- Mobile app usage

### RFM Analysis
- Recency (days since last purchase)
- Frequency (number of purchases)
- Monetary (total spending)
- RFM score (combined metric)

## Technical Details

### Clustering Algorithm
- **Algorithm**: K-means clustering
- **Optimization**: Elbow method + Silhouette score
- **Preprocessing**: StandardScaler normalization
- **Dimensionality Reduction**: PCA for visualization

### Model Performance
- Automatic optimal cluster detection
- Comprehensive evaluation metrics
- Robust feature scaling
- Missing value handling

## Future Enhancements

1. **Additional Algorithms**: DBSCAN, Hierarchical clustering
2. **Real-time Updates**: Streaming customer data processing
3. **A/B Testing**: Marketing strategy testing framework
4. **API Integration**: REST API for real-time segmentation
5. **Dashboard**: Web-based visualization dashboard

## Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- plotly>=5.17.0

## License

This project is part of the AI Data Engineering Portfolio. 