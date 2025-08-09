"""
Advanced Customer Segmentation with Multiple Algorithms
====================================================

This enhanced version includes:
- Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
- Model persistence and loading
- Enhanced evaluation metrics
- Cross-validation for clustering
- Advanced visualization techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class AdvancedCustomerSegmentation:
    def __init__(self):
        """Initialize the advanced customer segmentation model."""
        self.data = None
        self.scaled_data = None
        self.models = {}
        self.best_model = None
        self.best_algorithm = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.results = {}
        
    def generate_sample_data(self, n_customers=1000):
        """Generate enhanced sample customer data.

        Note: Reproducibility is controlled by the project's global seed
        via shared_utils.reproducibility.set_global_seed.
        """
        
        # Generate more diverse customer data
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
            'customer_satisfaction': np.random.normal(4, 0.5, n_customers),
            'loyalty_program': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
            'premium_member': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
            'returns_rate': np.random.beta(1, 10, n_customers),
            'review_count': np.random.poisson(3, n_customers),
            'avg_rating': np.random.normal(4.2, 0.3, n_customers)
        }
        
        self.data = pd.DataFrame(customer_data)
        
        # Ensure realistic values
        self.data['age'] = np.clip(np.abs(self.data['age']), 18, 80)
        self.data['income'] = np.clip(np.abs(self.data['income']), 20000, 150000)
        self.data['avg_order_value'] = np.clip(np.abs(self.data['avg_order_value']), 10, 500)
        self.data['days_since_last_purchase'] = np.clip(np.abs(self.data['days_since_last_purchase']), 1, 365)
        self.data['mobile_app_usage'] = np.clip(self.data['mobile_app_usage'], 0, 1)
        self.data['customer_satisfaction'] = np.clip(self.data['customer_satisfaction'], 1, 5)
        self.data['returns_rate'] = np.clip(self.data['returns_rate'], 0, 0.3)
        self.data['avg_rating'] = np.clip(self.data['avg_rating'], 1, 5)
        
        return self.data
    
    def add_advanced_features(self):
        """Add advanced RFM and behavioral features."""
        # Basic RFM
        self.data['recency'] = self.data['days_since_last_purchase']
        self.data['frequency'] = self.data['total_purchases']
        self.data['monetary'] = self.data['total_purchases'] * self.data['avg_order_value']
        
        # Advanced RFM scoring
        rfm_scores = pd.DataFrame()
        
        # Recency scoring (lower is better)
        rfm_scores['r_score'] = pd.qcut(self.data['recency'], q=5, labels=[5, 4, 3, 2, 1])
        
        # Frequency scoring (higher is better)
        rfm_scores['f_score'] = pd.qcut(self.data['frequency'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Monetary scoring (higher is better)
        rfm_scores['m_score'] = pd.qcut(self.data['monetary'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Combined RFM score
        self.data['rfm_score'] = (
            rfm_scores['r_score'].astype(int) * 100 + 
            rfm_scores['f_score'].astype(int) * 10 + 
            rfm_scores['m_score'].astype(int)
        )
        
        # Customer lifetime value (CLV)
        self.data['clv'] = (
            self.data['avg_order_value'] * 
            self.data['purchase_frequency'] * 
            (1 - self.data['returns_rate']) * 
            self.data['customer_satisfaction']
        )
        
        # Engagement score
        self.data['engagement_score'] = (
            self.data['online_visits'] * 0.3 +
            self.data['mobile_app_usage'] * 0.3 +
            self.data['review_count'] * 0.2 +
            self.data['loyalty_program'] * 0.2
        )
        
        # Risk score (for churn prediction)
        self.data['risk_score'] = (
            self.data['days_since_last_purchase'] * 0.4 +
            (1 - self.data['customer_satisfaction']) * 0.3 +
            self.data['returns_rate'] * 0.3
        )
        
        return self.data
    
    def prepare_features(self, features=None):
        """Prepare features with advanced preprocessing."""
        if features is None:
            features = [
                'age', 'income', 'spending_score', 'purchase_frequency',
                'avg_order_value', 'total_purchases', 'online_visits',
                'mobile_app_usage', 'customer_satisfaction', 'rfm_score',
                'clv', 'engagement_score', 'risk_score'
            ]
        
        feature_data = self.data[features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.mean())
        
        # Remove outliers using IQR method
        for column in feature_data.columns:
            Q1 = feature_data[column].quantile(0.25)
            Q3 = feature_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            feature_data[column] = np.clip(feature_data[column], lower_bound, upper_bound)
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(feature_data)
        
        return self.scaled_data, features
    
    def evaluate_clustering(self, model, algorithm_name):
        """Evaluate clustering performance using multiple metrics."""
        labels = model.labels_
        
        # Skip evaluation if all points are in one cluster
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': float('inf'),
                'n_clusters': 1
            }
        
        metrics = {
            'silhouette_score': silhouette_score(self.scaled_data, labels),
            'calinski_harabasz_score': calinski_harabasz_score(self.scaled_data, labels),
            'davies_bouldin_score': davies_bouldin_score(self.scaled_data, labels),
            'n_clusters': len(np.unique(labels))
        }
        
        return metrics
    
    def find_optimal_parameters(self, algorithm='kmeans', max_clusters=10):
        """Find optimal parameters for different clustering algorithms."""
        best_score = -1
        best_params = None
        best_model = None
        
        if algorithm == 'kmeans':
            param_grid = {
                'n_clusters': range(2, max_clusters + 1),
                'init': ['k-means++', 'random'],
                'n_init': [10, 20]
            }
            
            for params in ParameterGrid(param_grid):
                model = KMeans(**params, random_state=42)
                model.fit(self.scaled_data)
                
                metrics = self.evaluate_clustering(model, 'kmeans')
                score = metrics['silhouette_score']
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
        
        elif algorithm == 'dbscan':
            param_grid = {
                'eps': np.arange(0.1, 2.0, 0.1),
                'min_samples': [3, 5, 10, 15]
            }
            
            for params in ParameterGrid(param_grid):
                model = DBSCAN(**params)
                model.fit(self.scaled_data)
                
                metrics = self.evaluate_clustering(model, 'dbscan')
                score = metrics['silhouette_score']
                
                if score > best_score and metrics['n_clusters'] > 1:
                    best_score = score
                    best_params = params
                    best_model = model
        
        elif algorithm == 'hierarchical':
            param_grid = {
                'n_clusters': range(2, max_clusters + 1),
                'linkage': ['ward', 'complete', 'average']
            }
            
            for params in ParameterGrid(param_grid):
                model = AgglomerativeClustering(**params)
                model.fit(self.scaled_data)
                
                metrics = self.evaluate_clustering(model, 'hierarchical')
                score = metrics['silhouette_score']
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
        
        return best_model, best_params, best_score
    
    def compare_algorithms(self, max_clusters=10):
        """Compare different clustering algorithms."""
        algorithms = ['kmeans', 'dbscan', 'hierarchical']
        
        for algorithm in algorithms:
            print(f"Testing {algorithm.upper()} algorithm...")
            model, params, score = self.find_optimal_parameters(algorithm, max_clusters)
            
            if model is not None:
                self.models[algorithm] = {
                    'model': model,
                    'params': params,
                    'score': score,
                    'metrics': self.evaluate_clustering(model, algorithm)
                }
        
        # Find best overall algorithm
        best_algorithm = max(self.models.keys(), 
                           key=lambda x: self.models[x]['score'])
        
        self.best_algorithm = best_algorithm
        self.best_model = self.models[best_algorithm]['model']
        self.best_params = self.models[best_algorithm]['params']
        
        return self.models
    
    def visualize_algorithm_comparison(self, save_path=None, show=True):
        """Create comprehensive visualization comparing algorithms.

        Args:
            save_path (str | None): If provided, saves the interactive plot to this path (HTML).
            show (bool): If True, display the figure in an interactive window.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Algorithm Comparison', 'Silhouette Scores',
                          'Number of Clusters', 'Calinski-Harabasz Scores'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        algorithms = list(self.models.keys())
        silhouette_scores = [self.models[alg]['metrics']['silhouette_score'] 
                           for alg in algorithms]
        n_clusters = [self.models[alg]['metrics']['n_clusters'] 
                     for alg in algorithms]
        calinski_scores = [self.models[alg]['metrics']['calinski_harabasz_score'] 
                          for alg in algorithms]
        
        # Algorithm comparison
        fig.add_trace(
            go.Bar(x=algorithms, y=silhouette_scores, name='Silhouette Score'),
            row=1, col=1
        )
        
        # Silhouette scores
        fig.add_trace(
            go.Bar(x=algorithms, y=silhouette_scores, name='Silhouette'),
            row=1, col=2
        )
        
        # Number of clusters
        fig.add_trace(
            go.Bar(x=algorithms, y=n_clusters, name='Clusters'),
            row=2, col=1
        )
        
        # Calinski-Harabasz scores
        fig.add_trace(
            go.Scatter(x=algorithms, y=calinski_scores, mode='markers+lines',
                      name='Calinski-Harabasz'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Clustering Algorithm Comparison",
            height=600,
            showlegend=True
        )
        
        # Save and/or show plot
        if save_path is not None:
            try:
                fig.write_html(save_path)
            except Exception:
                pass
        if show and save_path is None:
            # Avoid double rendering when saving
            fig.show()
    
    def create_advanced_profiles(self, features):
        """Create advanced customer profiles with detailed insights."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        
        self.data['cluster'] = self.best_model.labels_
        profiles = {}
        
        for cluster_id in np.unique(self.best_model.labels_):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.data)) * 100,
                'characteristics': {},
                'insights': {},
                'recommendations': {}
            }
            
            # Calculate characteristics
            for feature in features:
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                profile['characteristics'][feature] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': cluster_data[feature].min(),
                    'max': cluster_data[feature].max()
                }
            
            # Generate insights
            profile['insights'] = self._generate_advanced_insights(cluster_data)
            
            # Generate recommendations
            profile['recommendations'] = self._generate_advanced_recommendations(cluster_data)
            
            profiles[cluster_id] = profile
        
        return profiles
    
    def _generate_advanced_insights(self, cluster_data):
        """Generate advanced insights for a cluster."""
        insights = {}
        
        # Value analysis
        avg_clv = cluster_data['clv'].mean()
        if avg_clv > cluster_data['clv'].quantile(0.75):
            insights['value_tier'] = 'High Value'
        elif avg_clv > cluster_data['clv'].quantile(0.5):
            insights['value_tier'] = 'Medium Value'
        else:
            insights['value_tier'] = 'Low Value'
        
        # Risk analysis
        avg_risk = cluster_data['risk_score'].mean()
        if avg_risk > cluster_data['risk_score'].quantile(0.75):
            insights['risk_level'] = 'High Risk'
        elif avg_risk > cluster_data['risk_score'].quantile(0.5):
            insights['risk_level'] = 'Medium Risk'
        else:
            insights['risk_level'] = 'Low Risk'
        
        # Engagement analysis
        avg_engagement = cluster_data['engagement_score'].mean()
        if avg_engagement > cluster_data['engagement_score'].quantile(0.75):
            insights['engagement_level'] = 'Highly Engaged'
        elif avg_engagement > cluster_data['engagement_score'].quantile(0.5):
            insights['engagement_level'] = 'Moderately Engaged'
        else:
            insights['engagement_level'] = 'Low Engagement'
        
        # Loyalty analysis
        loyalty_rate = cluster_data['loyalty_program'].mean()
        premium_rate = cluster_data['premium_member'].mean()
        
        insights['loyalty_rate'] = f"{loyalty_rate:.1%}"
        insights['premium_rate'] = f"{premium_rate:.1%}"
        
        return insights
    
    def _generate_advanced_recommendations(self, cluster_data):
        """Generate advanced recommendations for a cluster."""
        recommendations = {
            'targeting': [],
            'messaging': [],
            'offers': [],
            'channels': [],
            'retention_strategies': [],
            'upselling_opportunities': []
        }
        
        # Value-based recommendations
        avg_clv = cluster_data['clv'].mean()
        if avg_clv > cluster_data['clv'].quantile(0.75):
            recommendations['targeting'].append("Premium segment")
            recommendations['offers'].append("Exclusive products and VIP services")
            recommendations['upselling_opportunities'].append("Premium membership upgrade")
        elif avg_clv > cluster_data['clv'].quantile(0.5):
            recommendations['targeting'].append("Mid-market segment")
            recommendations['offers'].append("Value-based promotions")
            recommendations['upselling_opportunities'].append("Loyalty program enrollment")
        else:
            recommendations['targeting'].append("Budget-conscious segment")
            recommendations['offers'].append("Discounts and deals")
            recommendations['upselling_opportunities'].append("Basic loyalty program")
        
        # Risk-based retention strategies
        avg_risk = cluster_data['risk_score'].mean()
        if avg_risk > cluster_data['risk_score'].quantile(0.75):
            recommendations['retention_strategies'].append("Proactive outreach")
            recommendations['retention_strategies'].append("Special retention offers")
            recommendations['retention_strategies'].append("Personalized communication")
        
        # Engagement-based channels
        avg_engagement = cluster_data['engagement_score'].mean()
        if avg_engagement > cluster_data['engagement_score'].quantile(0.75):
            recommendations['channels'].append("Digital marketing")
            recommendations['channels'].append("Social media")
            recommendations['channels'].append("Mobile app")
        else:
            recommendations['channels'].append("Email campaigns")
            recommendations['channels'].append("Traditional marketing")
        
        return recommendations
    
    def save_model(self, filepath='customer_segmentation_model.pkl'):
        """Save the trained model and metadata."""
        model_data = {
            'best_model': self.best_model,
            'best_algorithm': self.best_algorithm,
            'best_params': self.best_params,
            'scaler': self.scaler,
            'pca': self.pca,
            'models': self.models,
            'data': self.data,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='customer_segmentation_model.pkl'):
        """Load a previously trained model."""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['best_model']
        self.best_algorithm = model_data['best_algorithm']
        self.best_params = model_data['best_params']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.models = model_data['models']
        self.data = model_data['data']
        
        print(f"Model loaded from {filepath}")
        print(f"Best algorithm: {self.best_algorithm}")
        print(f"Best parameters: {self.best_params}")

def main():
    """Main function to run advanced customer segmentation."""
    # Initialize segmentation model
    segmentation = AdvancedCustomerSegmentation()
    
    # Generate sample data
    print("Generating enhanced sample customer data...")
    data = segmentation.generate_sample_data(n_customers=1000)
    print(f"Generated {len(data)} customer records")
    
    # Add advanced features
    segmentation.add_advanced_features()
    
    # Prepare features
    features = [
        'age', 'income', 'spending_score', 'purchase_frequency',
        'avg_order_value', 'total_purchases', 'online_visits',
        'mobile_app_usage', 'customer_satisfaction', 'rfm_score',
        'clv', 'engagement_score', 'risk_score'
    ]
    
    scaled_data, feature_names = segmentation.prepare_features(features)
    print(f"Prepared {len(feature_names)} features for clustering")
    
    # Compare algorithms
    print("Comparing clustering algorithms...")
    models = segmentation.compare_algorithms(max_clusters=8)
    
    # Display results
    print("\nAlgorithm Comparison Results:")
    for algorithm, results in models.items():
        print(f"\n{algorithm.upper()}:")
        print(f"  Silhouette Score: {results['metrics']['silhouette_score']:.3f}")
        print(f"  Number of Clusters: {results['metrics']['n_clusters']}")
        print(f"  Parameters: {results['params']}")
    
    print(f"\nBest Algorithm: {segmentation.best_algorithm.upper()}")
    print(f"Best Silhouette Score: {models[segmentation.best_algorithm]['score']:.3f}")
    
    # Visualize comparison
    segmentation.visualize_algorithm_comparison()
    
    # Create advanced profiles
    profiles = segmentation.create_advanced_profiles(features)
    print("\nAdvanced Customer Profiles:")
    for cluster_id, profile in profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
        print(f"  Value Tier: {profile['insights']['value_tier']}")
        print(f"  Risk Level: {profile['insights']['risk_level']}")
        print(f"  Engagement: {profile['insights']['engagement_level']}")
        print(f"  Loyalty Rate: {profile['insights']['loyalty_rate']}")
        print(f"  Premium Rate: {profile['insights']['premium_rate']}")
        
        print("  Recommendations:")
        for category, recs in profile['recommendations'].items():
            if recs:
                print(f"    {category.title()}: {', '.join(recs)}")
    
    # Save model
    segmentation.save_model()
    
    print("\nAdvanced customer segmentation analysis completed!")

if __name__ == "__main__":
    main() 

# =============================
# Config-driven experiment entry
# =============================
def run_experiment(cfg):
    """Run the advanced segmentation experiment using a config-driven setup.

    Expected cfg structure (OmegaConf/Dict):
        cfg.task: "advanced_segmentation"
        cfg.mlflow.experiment_name: str
        cfg.advanced_segmentation.n_customers: int
        cfg.advanced_segmentation.max_clusters: int
        cfg.advanced_segmentation.features: List[str]
        cfg.advanced_segmentation.artifacts.model_path: str
        cfg.advanced_segmentation.artifacts.profiles_path: str
        cfg.advanced_segmentation.artifacts.comparison_plot_path: str
        cfg.advanced_segmentation.artifacts.models_summary_path: str
    """
    import os
    import json
    from pathlib import Path
    
    try:
        import mlflow
    except Exception:
        mlflow = None

    # Initialize model
    segmentation = AdvancedCustomerSegmentation()

    # Generate and enrich data
    n_customers = int(cfg.advanced_segmentation.get('n_customers', 1000))
    data = segmentation.generate_sample_data(n_customers=n_customers)
    segmentation.add_advanced_features()

    # Prepare features
    features = list(cfg.advanced_segmentation.get('features', [
        'age', 'income', 'spending_score', 'purchase_frequency',
        'avg_order_value', 'total_purchases', 'online_visits',
        'mobile_app_usage', 'customer_satisfaction', 'rfm_score',
        'clv', 'engagement_score', 'risk_score'
    ]))
    segmentation.prepare_features(features)

    # Train & compare algorithms
    max_clusters = int(cfg.advanced_segmentation.get('max_clusters', 8))
    models = segmentation.compare_algorithms(max_clusters=max_clusters)

    # Prepare output paths (relative to current working directory, Hydra-friendly)
    artifacts_cfg = cfg.advanced_segmentation.get('artifacts', {})
    model_path = Path(artifacts_cfg.get('model_path', 'customer_segmentation_model.pkl'))
    profiles_path = Path(artifacts_cfg.get('profiles_path', 'profiles.json'))
    plot_path = Path(artifacts_cfg.get('comparison_plot_path', 'algorithm_comparison.html'))
    models_summary_path = Path(artifacts_cfg.get('models_summary_path', 'models_summary.json'))

    # Create directories if needed
    for p in [model_path, profiles_path, plot_path, models_summary_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # Save visualization to artifact
    segmentation.visualize_algorithm_comparison(save_path=str(plot_path), show=False)

    # Create profiles & save
    profiles = segmentation.create_advanced_profiles(features)
    with open(profiles_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, default=str)

    # Save model artifact
    segmentation.save_model(filepath=str(model_path))

    # Save models comparison summary
    summary = {}
    for alg, info in models.items():
        summary[alg] = {
            'score': float(info.get('score', -1)),
            'params': info.get('params'),
            'metrics': {
                'silhouette_score': float(info['metrics'].get('silhouette_score', -1)),
                'calinski_harabasz_score': float(info['metrics'].get('calinski_harabasz_score', -1)),
                'davies_bouldin_score': float(info['metrics'].get('davies_bouldin_score', -1)),
                'n_clusters': int(info['metrics'].get('n_clusters', 0)),
            }
        }
    with open(models_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Optional: MLflow logging
    if mlflow is not None:
        try:
            # Set experiment if provided
            try:
                exp_name = cfg.get('mlflow', {}).get('experiment_name', None)
            except Exception:
                exp_name = None
            if exp_name:
                try:
                    mlflow.set_experiment(exp_name)
                except Exception:
                    pass

            run_name = f"advanced_segmentation_{segmentation.best_algorithm}"
            mlflow.start_run(run_name=run_name)

            # Params
            mlflow.log_param('n_customers', n_customers)
            mlflow.log_param('max_clusters', max_clusters)
            mlflow.log_param('features', ','.join(features))
            mlflow.log_param('best_algorithm', segmentation.best_algorithm)
            if segmentation.best_params is not None:
                for k, v in segmentation.best_params.items():
                    mlflow.log_param(f"best_params.{k}", v)

            # Metrics (best algorithm)
            best_metrics = models[segmentation.best_algorithm]['metrics']
            mlflow.log_metric('silhouette_score', float(best_metrics['silhouette_score']))
            mlflow.log_metric('calinski_harabasz_score', float(best_metrics['calinski_harabasz_score']))
            mlflow.log_metric('davies_bouldin_score', float(best_metrics['davies_bouldin_score']))
            mlflow.log_metric('n_clusters', int(best_metrics['n_clusters']))

            # Artifacts
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(profiles_path))
            mlflow.log_artifact(str(plot_path))
            mlflow.log_artifact(str(models_summary_path))
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass