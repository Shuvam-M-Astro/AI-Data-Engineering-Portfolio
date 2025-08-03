"""
Customer Segmentation Demo
=========================

This script demonstrates the key features of both basic and advanced
customer segmentation implementations.
"""

import time
from customer_segmentation import CustomerSegmentation
from advanced_segmentation import AdvancedCustomerSegmentation

def demo_basic_segmentation():
    """Demonstrate basic customer segmentation."""
    print("=" * 60)
    print("BASIC CUSTOMER SEGMENTATION DEMO")
    print("=" * 60)
    
    # Initialize
    segmentation = CustomerSegmentation()
    
    # Generate data
    print("1. Generating sample customer data...")
    data = segmentation.generate_sample_data(n_customers=500)
    print(f"   Generated {len(data)} customer records")
    
    # Add RFM features
    print("2. Adding RFM features...")
    segmentation.add_rfm_features()
    
    # Prepare features
    print("3. Preparing features for clustering...")
    features = [
        'age', 'income', 'spending_score', 'purchase_frequency',
        'avg_order_value', 'total_purchases', 'online_visits',
        'mobile_app_usage', 'customer_satisfaction', 'rfm_score'
    ]
    scaled_data, feature_names = segmentation.prepare_features(features)
    print(f"   Prepared {len(feature_names)} features")
    
    # Find optimal clusters
    print("4. Finding optimal number of clusters...")
    optimal_k = segmentation.find_optimal_clusters(max_clusters=6)
    print(f"   Optimal clusters: {optimal_k}")
    
    # Perform clustering
    print("5. Performing clustering...")
    model = segmentation.perform_clustering(n_clusters=optimal_k)
    
    # Analyze clusters
    print("6. Analyzing clusters...")
    cluster_analysis = segmentation.analyze_clusters(features)
    print("\nCluster Analysis Summary:")
    print(cluster_analysis[['cluster_size', 'cluster_percentage']].round(2))
    
    # Create profiles
    print("7. Creating customer profiles...")
    profiles = segmentation.create_customer_profiles(features)
    
    print("\nCustomer Profiles:")
    for cluster_id, profile in profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
        print(f"  Description: {profile['description']}")
    
    # Generate strategies
    print("8. Generating marketing strategies...")
    strategies = segmentation.recommend_strategies(profiles)
    
    print("\nMarketing Strategies:")
    for cluster_id, strategy in strategies.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Targeting: {', '.join(strategy['targeting'])}")
        print(f"  Messaging: {', '.join(strategy['messaging'])}")
        print(f"  Offers: {', '.join(strategy['offers'])}")
        print(f"  Channels: {', '.join(strategy['channels'])}")
    
    print("\nBasic segmentation demo completed!")
    return segmentation

def demo_advanced_segmentation():
    """Demonstrate advanced customer segmentation."""
    print("\n" + "=" * 60)
    print("ADVANCED CUSTOMER SEGMENTATION DEMO")
    print("=" * 60)
    
    # Initialize
    segmentation = AdvancedCustomerSegmentation()
    
    # Generate enhanced data
    print("1. Generating enhanced customer data...")
    data = segmentation.generate_sample_data(n_customers=500)
    print(f"   Generated {len(data)} customer records with advanced features")
    
    # Add advanced features
    print("2. Adding advanced features (CLV, engagement, risk scores)...")
    segmentation.add_advanced_features()
    
    # Prepare features
    print("3. Preparing advanced features for clustering...")
    features = [
        'age', 'income', 'spending_score', 'purchase_frequency',
        'avg_order_value', 'total_purchases', 'online_visits',
        'mobile_app_usage', 'customer_satisfaction', 'rfm_score',
        'clv', 'engagement_score', 'risk_score'
    ]
    scaled_data, feature_names = segmentation.prepare_features(features)
    print(f"   Prepared {len(feature_names)} advanced features")
    
    # Compare algorithms
    print("4. Comparing clustering algorithms...")
    start_time = time.time()
    models = segmentation.compare_algorithms(max_clusters=6)
    comparison_time = time.time() - start_time
    print(f"   Algorithm comparison completed in {comparison_time:.2f} seconds")
    
    # Display results
    print("\nAlgorithm Comparison Results:")
    for algorithm, results in models.items():
        print(f"\n{algorithm.upper()}:")
        print(f"  Silhouette Score: {results['metrics']['silhouette_score']:.3f}")
        print(f"  Number of Clusters: {results['metrics']['n_clusters']}")
        print(f"  Parameters: {results['params']}")
    
    print(f"\nBest Algorithm: {segmentation.best_algorithm.upper()}")
    print(f"Best Silhouette Score: {models[segmentation.best_algorithm]['score']:.3f}")
    
    # Create advanced profiles
    print("5. Creating advanced customer profiles...")
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
    print("6. Saving model for future use...")
    segmentation.save_model('demo_advanced_model.pkl')
    print("   Model saved successfully!")
    
    print("\nAdvanced segmentation demo completed!")
    return segmentation

def compare_approaches():
    """Compare basic vs advanced approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: BASIC vs ADVANCED APPROACHES")
    print("=" * 60)
    
    print("\nBasic Approach Features:")
    print("✓ K-means clustering")
    print("✓ RFM analysis")
    print("✓ Elbow method for optimal clusters")
    print("✓ Basic customer profiles")
    print("✓ Simple marketing strategies")
    
    print("\nAdvanced Approach Features:")
    print("✓ Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)")
    print("✓ Advanced RFM scoring")
    print("✓ Customer Lifetime Value (CLV)")
    print("✓ Engagement and risk scoring")
    print("✓ Comprehensive algorithm comparison")
    print("✓ Advanced customer insights")
    print("✓ Detailed marketing recommendations")
    print("✓ Model persistence")
    print("✓ Enhanced visualizations")
    
    print("\nUse Cases:")
    print("Basic: Quick customer segmentation for small datasets")
    print("Advanced: Production-ready segmentation with multiple algorithms")

def main():
    """Run the complete demo."""
    print("CUSTOMER SEGMENTATION DEMO")
    print("=" * 60)
    
    try:
        # Run basic demo
        basic_seg = demo_basic_segmentation()
        
        # Run advanced demo
        advanced_seg = demo_advanced_segmentation()
        
        # Compare approaches
        compare_approaches()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 'python customer_segmentation.py' for basic analysis")
        print("2. Run 'python advanced_segmentation.py' for advanced analysis")
        print("3. Run 'python test_segmentation.py' to validate functionality")
        print("4. Check the README.md for detailed documentation")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Please check your installation and dependencies.")

if __name__ == "__main__":
    main() 