"""
Test script for Customer Segmentation
====================================

This script tests the functionality of both basic and advanced
customer segmentation implementations.
"""

import unittest
import numpy as np
import pandas as pd
from customer_segmentation import CustomerSegmentation
from advanced_segmentation import AdvancedCustomerSegmentation

class TestCustomerSegmentation(unittest.TestCase):
    """Test cases for basic customer segmentation."""
    
    def setUp(self):
        """Set up test data."""
        self.segmentation = CustomerSegmentation()
        self.data = self.segmentation.generate_sample_data(n_customers=100)
        self.segmentation.add_rfm_features()
    
    def test_data_generation(self):
        """Test data generation functionality."""
        self.assertIsNotNone(self.data)
        self.assertEqual(len(self.data), 100)
        self.assertIn('customer_id', self.data.columns)
        self.assertIn('age', self.data.columns)
        self.assertIn('income', self.data.columns)
    
    def test_rfm_features(self):
        """Test RFM feature generation."""
        self.assertIn('recency', self.data.columns)
        self.assertIn('frequency', self.data.columns)
        self.assertIn('monetary', self.data.columns)
        self.assertIn('rfm_score', self.data.columns)
        
        # Check that RFM scores are reasonable
        self.assertTrue(all(self.data['recency'] >= 0))
        self.assertTrue(all(self.data['frequency'] >= 0))
        self.assertTrue(all(self.data['monetary'] >= 0))
    
    def test_feature_preparation(self):
        """Test feature preparation."""
        features = ['age', 'income', 'spending_score', 'rfm_score']
        scaled_data, feature_names = self.segmentation.prepare_features(features)
        
        self.assertEqual(scaled_data.shape[1], len(features))
        self.assertEqual(len(feature_names), len(features))
        self.assertIsNotNone(scaled_data)
    
    def test_clustering(self):
        """Test clustering functionality."""
        features = ['age', 'income', 'spending_score', 'rfm_score']
        scaled_data, feature_names = self.segmentation.prepare_features(features)
        
        # Test clustering with 3 clusters
        model = self.segmentation.perform_clustering(n_clusters=3)
        self.assertIsNotNone(model)
        self.assertEqual(len(np.unique(self.segmentation.data['cluster'])), 3)
    
    def test_profile_creation(self):
        """Test customer profile creation."""
        features = ['age', 'income', 'spending_score', 'rfm_score']
        scaled_data, feature_names = self.segmentation.prepare_features(features)
        self.segmentation.perform_clustering(n_clusters=3)
        
        profiles = self.segmentation.create_customer_profiles(features)
        self.assertIsNotNone(profiles)
        self.assertEqual(len(profiles), 3)
        
        for cluster_id, profile in profiles.items():
            self.assertIn('size', profile)
            self.assertIn('percentage', profile)
            self.assertIn('description', profile)

class TestAdvancedCustomerSegmentation(unittest.TestCase):
    """Test cases for advanced customer segmentation."""
    
    def setUp(self):
        """Set up test data."""
        self.segmentation = AdvancedCustomerSegmentation()
        self.data = self.segmentation.generate_sample_data(n_customers=100)
        self.segmentation.add_advanced_features()
    
    def test_advanced_data_generation(self):
        """Test advanced data generation."""
        self.assertIsNotNone(self.data)
        self.assertEqual(len(self.data), 100)
        
        # Check for advanced features
        advanced_features = ['loyalty_program', 'premium_member', 'returns_rate', 
                           'review_count', 'avg_rating']
        for feature in advanced_features:
            self.assertIn(feature, self.data.columns)
    
    def test_advanced_features(self):
        """Test advanced feature generation."""
        self.assertIn('clv', self.data.columns)
        self.assertIn('engagement_score', self.data.columns)
        self.assertIn('risk_score', self.data.columns)
        
        # Check that advanced scores are reasonable
        self.assertTrue(all(self.data['clv'] >= 0))
        self.assertTrue(all(self.data['engagement_score'] >= 0))
        self.assertTrue(all(self.data['risk_score'] >= 0))
    
    def test_algorithm_comparison(self):
        """Test algorithm comparison functionality."""
        features = ['age', 'income', 'spending_score', 'rfm_score', 'clv']
        scaled_data, feature_names = self.segmentation.prepare_features(features)
        
        # Test with smaller parameter space for speed
        models = self.segmentation.compare_algorithms(max_clusters=4)
        
        self.assertIsNotNone(models)
        self.assertIn('kmeans', models)
        self.assertIsNotNone(self.segmentation.best_algorithm)
        self.assertIsNotNone(self.segmentation.best_model)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        features = ['age', 'income', 'spending_score', 'rfm_score', 'clv']
        scaled_data, feature_names = self.segmentation.prepare_features(features)
        self.segmentation.compare_algorithms(max_clusters=4)
        
        # Save model
        self.segmentation.save_model('test_model.pkl')
        
        # Create new instance and load model
        new_segmentation = AdvancedCustomerSegmentation()
        new_segmentation.load_model('test_model.pkl')
        
        self.assertEqual(new_segmentation.best_algorithm, self.segmentation.best_algorithm)
        self.assertIsNotNone(new_segmentation.best_model)
    
    def test_advanced_profiles(self):
        """Test advanced profile creation."""
        features = ['age', 'income', 'spending_score', 'rfm_score', 'clv']
        scaled_data, feature_names = self.segmentation.prepare_features(features)
        self.segmentation.compare_algorithms(max_clusters=4)
        
        profiles = self.segmentation.create_advanced_profiles(features)
        self.assertIsNotNone(profiles)
        
        for cluster_id, profile in profiles.items():
            self.assertIn('insights', profile)
            self.assertIn('recommendations', profile)
            self.assertIn('value_tier', profile['insights'])
            self.assertIn('risk_level', profile['insights'])

def run_performance_test():
    """Run a performance test with larger dataset."""
    print("Running performance test...")
    
    # Test basic segmentation
    basic_seg = CustomerSegmentation()
    data = basic_seg.generate_sample_data(n_customers=1000)
    basic_seg.add_rfm_features()
    
    features = ['age', 'income', 'spending_score', 'rfm_score']
    scaled_data, feature_names = basic_seg.prepare_features(features)
    
    import time
    start_time = time.time()
    basic_seg.perform_clustering(n_clusters=4)
    basic_time = time.time() - start_time
    
    print(f"Basic segmentation completed in {basic_time:.2f} seconds")
    
    # Test advanced segmentation
    adv_seg = AdvancedCustomerSegmentation()
    data = adv_seg.generate_sample_data(n_customers=1000)
    adv_seg.add_advanced_features()
    
    features = ['age', 'income', 'spending_score', 'rfm_score', 'clv']
    scaled_data, feature_names = adv_seg.prepare_features(features)
    
    start_time = time.time()
    adv_seg.compare_algorithms(max_clusters=6)
    adv_time = time.time() - start_time
    
    print(f"Advanced segmentation completed in {adv_time:.2f} seconds")
    
    return basic_time, adv_time

def main():
    """Run all tests."""
    print("Running customer segmentation tests...")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance test
    print("\n" + "="*50)
    basic_time, adv_time = run_performance_test()
    
    print(f"\nPerformance Summary:")
    print(f"Basic segmentation: {basic_time:.2f}s")
    print(f"Advanced segmentation: {adv_time:.2f}s")
    print(f"Speed difference: {adv_time/basic_time:.1f}x slower (expected due to multiple algorithms)")

if __name__ == "__main__":
    main() 