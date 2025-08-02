"""
Customer Segmentation using K-means Clustering
=============================================

This project implements customer segmentation using K-means clustering
to identify distinct customer groups based on their behavior patterns.

Features:
- Data preprocessing and feature engineering
- K-means clustering with elbow method
- Customer profile analysis
- Visualization and insights
- RFM analysis integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self):
        """Initialize the customer segmentation model."""
        self.data = None
        self.scaled_data = None
        self.kmeans_model = None
        self.n_clusters = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def generate_sample_data(self, n_customers=1000):
        """
        Generate sample customer data for demonstration.
        
        Args:
            n_customers (int): Number of customers to generate
        """
        np.random.seed(42)
        
        # Generate customer data
        customer_data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(35, 12, n_customers),
            'income': np.random.normal(50000, 20000, n_customers),
            'spending_score': np.random.normal(50, 20, n_customers),
            'purchase_frequency': np.random.poisson(5, n_customers),
            'avg_order_value': np.random.normal(100, 30, n_customers),
            'days_since_last_purchase': np.random.exponential(30, n_customers),
            'total_purchases': np.random.poisson(20, n_customers),
            'online_visits': np.random.poisson(15, n_customers),
            'mobile_app_usage': np.random.beta(2, 5, n_customers),
            'customer_satisfaction': np.random.normal(4, 0.5, n_customers)
        }
        
        self.data = pd.DataFrame(customer_data)
        
        # Ensure positive values
        self.data['age'] = np.abs(self.data['age'])
        self.data['income'] = np.abs(self.data['income'])
        self.data['avg_order_value'] = np.abs(self.data['avg_order_value'])
        self.data['days_since_last_purchase'] = np.abs(self.data['days_since_last_purchase'])
        self.data['mobile_app_usage'] = np.clip(self.data['mobile_app_usage'], 0, 1)
        self.data['customer_satisfaction'] = np.clip(self.data['customer_satisfaction'], 1, 5)
        
        return self.data
    
    def add_rfm_features(self):
        """Add RFM (Recency, Frequency, Monetary) features."""
        # Recency (days since last purchase)
        self.data['recency'] = self.data['days_since_last_purchase']
        
        # Frequency (number of purchases)
        self.data['frequency'] = self.data['total_purchases']
        
        # Monetary (total spending)
        self.data['monetary'] = self.data['total_purchases'] * self.data['avg_order_value']
        
        # RFM Score (combined metric)
        self.data['rfm_score'] = (
            (1 / (1 + self.data['recency'])) * 
            self.data['frequency'] * 
            np.log(self.data['monetary'] + 1)
        )
        
        return self.data
    
    def prepare_features(self, features=None):
        """
        Prepare features for clustering.
        
        Args:
            features (list): List of features to use for clustering
        """
        if features is None:
            features = [
                'age', 'income', 'spending_score', 'purchase_frequency',
                'avg_order_value', 'total_purchases', 'online_visits',
                'mobile_app_usage', 'customer_satisfaction', 'rfm_score'
            ]
        
        # Select features
        feature_data = self.data[features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.mean())
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(feature_data)
        
        return self.scaled_data, features
    
    def find_optimal_clusters(self, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            max_clusters (int): Maximum number of clusters to test
        """
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))
        
        # Plot elbow curve
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(K_range, silhouette_scores, 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def perform_clustering(self, n_clusters=4):
        """
        Perform K-means clustering.
        
        Args:
            n_clusters (int): Number of clusters
        """
        self.n_clusters = n_clusters
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data['cluster'] = self.kmeans_model.fit_predict(self.scaled_data)
        
        return self.kmeans_model
    
    def analyze_clusters(self, features):
        """Analyze cluster characteristics."""
        cluster_analysis = self.data.groupby('cluster')[features].mean()
        
        # Add cluster sizes
        cluster_sizes = self.data['cluster'].value_counts().sort_index()
        cluster_analysis['cluster_size'] = cluster_sizes.values
        cluster_analysis['cluster_percentage'] = (cluster_sizes.values / len(self.data)) * 100
        
        return cluster_analysis
    
    def visualize_clusters(self, features):
        """Create comprehensive cluster visualizations."""
        # PCA for dimensionality reduction
        pca_data = self.pca.fit_transform(self.scaled_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA Cluster Visualization', 'Cluster Size Distribution', 
                          'Feature Importance by Cluster', 'RFM Analysis'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter3d"}]]
        )
        
        # Plot 1: PCA visualization
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i in range(self.n_clusters):
            mask = self.data['cluster'] == i
            fig.add_trace(
                go.Scatter(
                    x=pca_data[mask, 0],
                    y=pca_data[mask, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(color=colors[i % len(colors)], size=8)
                ),
                row=1, col=1
            )
        
        # Plot 2: Cluster size distribution
        cluster_counts = self.data['cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in cluster_counts.index],
                y=cluster_counts.values,
                name='Cluster Size',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Plot 3: Feature importance heatmap
        cluster_means = self.data.groupby('cluster')[features].mean()
        fig.add_trace(
            go.Heatmap(
                z=cluster_means.values,
                x=features,
                y=[f'Cluster {i}' for i in cluster_means.index],
                colorscale='Viridis',
                name='Feature Values'
            ),
            row=2, col=1
        )
        
        # Plot 4: RFM 3D scatter
        fig.add_trace(
            go.Scatter3d(
                x=self.data['recency'],
                y=self.data['frequency'],
                z=self.data['monetary'],
                mode='markers',
                marker=dict(
                    color=self.data['cluster'],
                    size=5,
                    colorscale='Viridis'
                ),
                name='RFM Analysis'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Customer Segmentation Analysis",
            height=800,
            showlegend=True
        )
        
        fig.show()
    
    def create_customer_profiles(self, features):
        """Create detailed customer profiles for each cluster."""
        cluster_analysis = self.analyze_clusters(features)
        
        profiles = {}
        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.data)) * 100,
                'characteristics': {}
            }
            
            # Analyze key characteristics
            for feature in features:
                mean_val = cluster_data[feature].mean()
                profile['characteristics'][feature] = mean_val
            
            # Create profile description
            profile['description'] = self._generate_profile_description(cluster_data, features)
            
            profiles[cluster_id] = profile
        
        return profiles
    
    def _generate_profile_description(self, cluster_data, features):
        """Generate human-readable profile description."""
        descriptions = []
        
        # Age analysis
        avg_age = cluster_data['age'].mean()
        if avg_age < 30:
            descriptions.append("Young customers")
        elif avg_age < 50:
            descriptions.append("Middle-aged customers")
        else:
            descriptions.append("Senior customers")
        
        # Income analysis
        avg_income = cluster_data['income'].mean()
        if avg_income > 70000:
            descriptions.append("High-income")
        elif avg_income > 40000:
            descriptions.append("Middle-income")
        else:
            descriptions.append("Lower-income")
        
        # Spending behavior
        avg_spending = cluster_data['spending_score'].mean()
        if avg_spending > 60:
            descriptions.append("High spenders")
        elif avg_spending > 40:
            descriptions.append("Moderate spenders")
        else:
            descriptions.append("Low spenders")
        
        # Engagement level
        avg_visits = cluster_data['online_visits'].mean()
        if avg_visits > 20:
            descriptions.append("Highly engaged")
        elif avg_visits > 10:
            descriptions.append("Moderately engaged")
        else:
            descriptions.append("Low engagement")
        
        return " | ".join(descriptions)
    
    def recommend_strategies(self, profiles):
        """Generate marketing strategies for each cluster."""
        strategies = {}
        
        for cluster_id, profile in profiles.items():
            strategy = {
                'cluster_id': cluster_id,
                'targeting': [],
                'messaging': [],
                'offers': [],
                'channels': []
            }
            
            # Analyze characteristics and recommend strategies
            characteristics = profile['characteristics']
            
            # Income-based targeting
            if characteristics.get('income', 0) > 70000:
                strategy['targeting'].append("Premium segment")
                strategy['offers'].append("Exclusive products and services")
                strategy['messaging'].append("Luxury and quality focus")
            elif characteristics.get('income', 0) > 40000:
                strategy['targeting'].append("Mid-market segment")
                strategy['offers'].append("Value-based promotions")
                strategy['messaging'].append("Quality and value balance")
            else:
                strategy['targeting'].append("Budget-conscious segment")
                strategy['offers'].append("Discounts and deals")
                strategy['messaging'].append("Affordability focus")
            
            # Engagement-based channels
            if characteristics.get('online_visits', 0) > 20:
                strategy['channels'].append("Digital marketing")
                strategy['channels'].append("Social media")
            else:
                strategy['channels'].append("Traditional marketing")
                strategy['channels'].append("Email campaigns")
            
            # Age-based messaging
            avg_age = characteristics.get('age', 35)
            if avg_age < 30:
                strategy['messaging'].append("Trendy and modern")
                strategy['channels'].append("Social media platforms")
            elif avg_age < 50:
                strategy['messaging'].append("Family and lifestyle")
                strategy['channels'].append("Content marketing")
            else:
                strategy['messaging'].append("Reliability and trust")
                strategy['channels'].append("Direct mail")
            
            strategies[cluster_id] = strategy
        
        return strategies

def main():
    """Main function to run customer segmentation."""
    # Initialize segmentation model
    segmentation = CustomerSegmentation()
    
    # Generate sample data
    print("Generating sample customer data...")
    data = segmentation.generate_sample_data(n_customers=1000)
    print(f"Generated {len(data)} customer records")
    
    # Add RFM features
    segmentation.add_rfm_features()
    
    # Prepare features for clustering
    features = [
        'age', 'income', 'spending_score', 'purchase_frequency',
        'avg_order_value', 'total_purchases', 'online_visits',
        'mobile_app_usage', 'customer_satisfaction', 'rfm_score'
    ]
    
    scaled_data, feature_names = segmentation.prepare_features(features)
    print(f"Prepared {len(feature_names)} features for clustering")
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    optimal_k = segmentation.find_optimal_clusters(max_clusters=8)
    
    # Perform clustering
    print(f"Performing clustering with {optimal_k} clusters...")
    segmentation.perform_clustering(n_clusters=optimal_k)
    
    # Analyze clusters
    cluster_analysis = segmentation.analyze_clusters(features)
    print("\nCluster Analysis:")
    print(cluster_analysis)
    
    # Visualize clusters
    print("Creating visualizations...")
    segmentation.visualize_clusters(features)
    
    # Create customer profiles
    profiles = segmentation.create_customer_profiles(features)
    print("\nCustomer Profiles:")
    for cluster_id, profile in profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
        print(f"  Description: {profile['description']}")
    
    # Generate marketing strategies
    strategies = segmentation.recommend_strategies(profiles)
    print("\nMarketing Strategies:")
    for cluster_id, strategy in strategies.items():
        print(f"\nCluster {cluster_id} Strategy:")
        print(f"  Targeting: {', '.join(strategy['targeting'])}")
        print(f"  Messaging: {', '.join(strategy['messaging'])}")
        print(f"  Offers: {', '.join(strategy['offers'])}")
        print(f"  Channels: {', '.join(strategy['channels'])}")

if __name__ == "__main__":
    main() 