"""
Financial Fraud Detection System
===============================

This module implements a comprehensive financial fraud detection system that:
- Analyzes transaction data for suspicious patterns
- Detects various types of financial fraud (credit card, insurance, etc.)
- Uses multiple ML algorithms for robust detection
- Provides real-time fraud scoring
- Implements explainable AI for fraud detection decisions

Author: AI Data Engineering Portfolio
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """
    Comprehensive Financial Fraud Detection System
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.fraud_threshold = 0.5
        
    def generate_synthetic_fraud_data(self, n_samples=50000):
        """
        Generate synthetic financial transaction data with fraud patterns
        
        Parameters:
        -----------
        n_samples : int
            Number of transactions to generate
            
        Returns:
        --------
        pd.DataFrame
            Synthetic transaction data with fraud labels
        """
        np.random.seed(42)
        
        # Transaction IDs
        transaction_ids = range(1, n_samples + 1)
        
        # Transaction amounts
        amounts = np.random.exponential(100, n_samples)
        amounts = np.clip(amounts, 1, 10000)
        
        # Merchant categories
        merchant_categories = np.random.choice([
            'retail', 'restaurant', 'gas_station', 'online_shopping',
            'travel', 'entertainment', 'healthcare', 'utilities'
        ], n_samples)
        
        # Transaction types
        transaction_types = np.random.choice([
            'purchase', 'withdrawal', 'transfer', 'payment'
        ], n_samples)
        
        # Time features
        hours = np.random.randint(0, 24, n_samples)
        days = np.random.randint(1, 32, n_samples)
        months = np.random.randint(1, 13, n_samples)
        
        # Location features
        countries = np.random.choice(['US', 'CA', 'UK', 'DE', 'FR', 'AU'], n_samples)
        cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples)
        
        # Customer features
        customer_ids = np.random.randint(1, 10001, n_samples)
        customer_ages = np.random.normal(45, 15, n_samples)
        customer_ages = np.clip(customer_ages, 18, 80)
        
        # Account features
        account_balances = np.random.exponential(5000, n_samples)
        account_ages = np.random.exponential(5, n_samples)
        
        # Create fraud patterns
        fraud_probability = np.zeros(n_samples)
        
        # Pattern 1: High amount transactions
        high_amount_mask = amounts > 2000
        fraud_probability[high_amount_mask] += 0.3
        
        # Pattern 2: Unusual hours (late night transactions)
        late_night_mask = (hours >= 23) | (hours <= 4)
        fraud_probability[late_night_mask] += 0.2
        
        # Pattern 3: International transactions
        international_mask = countries != 'US'
        fraud_probability[international_mask] += 0.15
        
        # Pattern 4: New accounts
        new_account_mask = account_ages < 1
        fraud_probability[new_account_mask] += 0.25
        
        # Pattern 5: Multiple transactions in short time
        # Simulate by adding noise to some transactions
        rapid_transaction_mask = np.random.random(n_samples) < 0.1
        fraud_probability[rapid_transaction_mask] += 0.4
        
        # Pattern 6: Unusual merchant categories
        unusual_merchant_mask = merchant_categories.isin(['online_shopping', 'travel'])
        fraud_probability[unusual_merchant_mask] += 0.1
        
        # Generate fraud labels
        fraud_labels = np.random.random(n_samples) < fraud_probability
        fraud_labels = fraud_labels.astype(int)
        
        # Add some random fraud for variety
        random_fraud_mask = np.random.random(n_samples) < 0.02
        fraud_labels[random_fraud_mask] = 1
        
        # Create additional features
        amount_log = np.log(amounts + 1)
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        
        # Create DataFrame
        data = pd.DataFrame({
            'transaction_id': transaction_ids,
            'amount': amounts,
            'amount_log': amount_log,
            'merchant_category': merchant_categories,
            'transaction_type': transaction_types,
            'hour': hours,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day': days,
            'month': months,
            'country': countries,
            'city': cities,
            'customer_id': customer_ids,
            'customer_age': customer_ages,
            'account_balance': account_balances,
            'account_age': account_ages,
            'is_fraud': fraud_labels
        })
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess transaction data for fraud detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw transaction data
            
        Returns:
        --------
        tuple
            (X, y) preprocessed features and target
        """
        # Select features for modeling
        categorical_features = ['merchant_category', 'transaction_type', 'country', 'city']
        numerical_features = ['amount', 'amount_log', 'hour_sin', 'hour_cos', 'day', 'month',
                            'customer_age', 'account_balance', 'account_age']
        
        # Encode categorical features
        X_encoded = data[numerical_features].copy()
        
        for feature in categorical_features:
            le = LabelEncoder()
            X_encoded[feature] = le.fit_transform(data[feature])
            self.label_encoders[feature] = le
        
        # Target variable
        y = data['is_fraud'].copy()
        
        # Handle missing values
        X_encoded = X_encoded.fillna(X_encoded.mean())
        
        return X_encoded, y
    
    def train_fraud_models(self, X, y):
        """
        Train multiple fraud detection models
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (fraud labels)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'SVM': SVC(
                probability=True, random_state=42, kernel='rbf'
            ),
            'Isolation Forest': IsolationForest(
                contamination=0.1, random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Isolation Forest':
                # Isolation Forest is unsupervised, so we use it differently
                model.fit(X_train_scaled)
                predictions = model.predict(X_train_scaled)
                # Convert to binary (1 for normal, -1 for anomaly)
                predictions = (predictions == -1).astype(int)
                y_pred = model.predict(X_test_scaled)
                y_pred = (y_pred == -1).astype(int)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            precision = (y_pred & y_test).sum() / max(y_pred.sum(), 1)
            recall = (y_pred & y_test).sum() / max(y_test.sum(), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            # ROC AUC for supervised models
            if name != 'Isolation Forest':
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5
            else:
                auc = 0.5
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            print(f"AUC: {auc:.3f}")
        
        self.models = results
        
        # Store feature importance from Random Forest
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return X_test_scaled, y_test
    
    def detect_fraud(self, transaction_data):
        """
        Detect fraud in new transaction data
        
        Parameters:
        -----------
        transaction_data : pd.DataFrame
            New transaction data to analyze
            
        Returns:
        --------
        dict
            Fraud detection results and scores
        """
        if not self.models:
            raise ValueError("Models must be trained first")
        
        # Preprocess new data
        categorical_features = ['merchant_category', 'transaction_type', 'country', 'city']
        numerical_features = ['amount', 'amount_log', 'hour_sin', 'hour_cos', 'day', 'month',
                            'customer_age', 'account_balance', 'account_age']
        
        # Encode categorical features
        X_new = transaction_data[numerical_features].copy()
        
        for feature in categorical_features:
            if feature in self.label_encoders:
                le = self.label_encoders[feature]
                # Handle new categories by assigning them to a default value
                new_categories = set(transaction_data[feature]) - set(le.classes_)
                if new_categories:
                    print(f"Warning: New categories found in {feature}: {new_categories}")
                X_new[feature] = transaction_data[feature].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Handle missing values
        X_new = X_new.fillna(X_new.mean())
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new)
        
        # Get predictions from all models
        fraud_scores = {}
        fraud_predictions = {}
        
        for name, result in self.models.items():
            model = result['model']
            
            if name == 'Isolation Forest':
                predictions = model.predict(X_new_scaled)
                predictions = (predictions == -1).astype(int)
                scores = model.decision_function(X_new_scaled)
            else:
                try:
                    predictions = model.predict(X_new_scaled)
                    scores = model.predict_proba(X_new_scaled)[:, 1]
                except:
                    predictions = np.zeros(len(X_new_scaled))
                    scores = np.zeros(len(X_new_scaled))
            
            fraud_scores[name] = scores
            fraud_predictions[name] = predictions
        
        # Ensemble prediction
        ensemble_scores = np.mean([scores for scores in fraud_scores.values()], axis=0)
        ensemble_predictions = (ensemble_scores > self.fraud_threshold).astype(int)
        
        return {
            'individual_scores': fraud_scores,
            'individual_predictions': fraud_predictions,
            'ensemble_scores': ensemble_scores,
            'ensemble_predictions': ensemble_predictions
        }
    
    def analyze_fraud_patterns(self, data):
        """
        Analyze patterns in fraud data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Transaction data with fraud labels
        """
        print("\nFraud Pattern Analysis")
        print("=" * 50)
        
        # Overall fraud rate
        fraud_rate = data['is_fraud'].mean()
        print(f"Overall fraud rate: {fraud_rate:.3f} ({fraud_rate*100:.1f}%)")
        
        # Fraud by amount
        print("\nFraud by Transaction Amount:")
        amount_bins = [0, 100, 500, 1000, 5000, float('inf')]
        amount_labels = ['$0-100', '$100-500', '$500-1000', '$1000-5000', '$5000+']
        data['amount_bin'] = pd.cut(data['amount'], bins=amount_bins, labels=amount_labels)
        
        fraud_by_amount = data.groupby('amount_bin')['is_fraud'].agg(['mean', 'count'])
        print(fraud_by_amount)
        
        # Fraud by hour
        print("\nFraud by Hour of Day:")
        fraud_by_hour = data.groupby('hour')['is_fraud'].mean()
        print(fraud_by_hour)
        
        # Fraud by merchant category
        print("\nFraud by Merchant Category:")
        fraud_by_merchant = data.groupby('merchant_category')['is_fraud'].agg(['mean', 'count'])
        print(fraud_by_merchant)
        
        # Fraud by country
        print("\nFraud by Country:")
        fraud_by_country = data.groupby('country')['is_fraud'].agg(['mean', 'count'])
        print(fraud_by_country)
    
    def visualize_fraud_analysis(self, data, predictions=None):
        """
        Create visualizations for fraud analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Transaction data
        predictions : dict, optional
            Model predictions
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Financial Fraud Detection Analysis', fontsize=16)
        
        # Fraud rate over time
        fraud_by_hour = data.groupby('hour')['is_fraud'].mean()
        axes[0, 0].plot(fraud_by_hour.index, fraud_by_hour.values, marker='o')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Fraud Rate')
        axes[0, 0].set_title('Fraud Rate by Hour')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Amount distribution
        axes[0, 1].hist(data[data['is_fraud'] == 0]['amount'], 
                        bins=50, alpha=0.7, label='Legitimate', density=True)
        axes[0, 1].hist(data[data['is_fraud'] == 1]['amount'], 
                        bins=50, alpha=0.7, label='Fraudulent', density=True)
        axes[0, 1].set_xlabel('Transaction Amount')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Amount Distribution')
        axes[0, 1].legend()
        axes[0, 1].set_xscale('log')
        
        # Fraud by merchant category
        fraud_by_merchant = data.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False)
        axes[0, 2].barh(fraud_by_merchant.index, fraud_by_merchant.values)
        axes[0, 2].set_xlabel('Fraud Rate')
        axes[0, 2].set_title('Fraud Rate by Merchant Category')
        
        # Age vs Fraud
        axes[1, 0].scatter(data[data['is_fraud'] == 0]['customer_age'], 
                           data[data['is_fraud'] == 0]['amount'], 
                           alpha=0.6, label='Legitimate', s=20)
        axes[1, 0].scatter(data[data['is_fraud'] == 1]['customer_age'], 
                           data[data['is_fraud'] == 1]['amount'], 
                           alpha=0.8, label='Fraudulent', s=50, color='red')
        axes[1, 0].set_xlabel('Customer Age')
        axes[1, 0].set_ylabel('Transaction Amount')
        axes[1, 0].set_title('Age vs Amount')
        axes[1, 0].legend()
        
        # Feature importance
        if self.feature_importance is not None:
            axes[1, 1].barh(self.feature_importance['feature'], 
                           self.feature_importance['importance'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Feature Importance')
        
        # Model comparison
        if predictions and 'ensemble_scores' in predictions:
            # ROC curve for ensemble
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(data['is_fraud'], predictions['ensemble_scores'])
            axes[1, 2].plot(fpr, tpr, label='Ensemble Model')
            axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 2].set_xlabel('False Positive Rate')
            axes[1, 2].set_ylabel('True Positive Rate')
            axes[1, 2].set_title('ROC Curve')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_fraud_report(self, transaction_id, transaction_data, fraud_results):
        """
        Generate a detailed fraud analysis report
        
        Parameters:
        -----------
        transaction_id : int
            Transaction identifier
        transaction_data : pd.DataFrame
            Transaction details
        fraud_results : dict
            Fraud detection results
            
        Returns:
        --------
        str
            Formatted fraud report
        """
        ensemble_score = fraud_results['ensemble_scores'][0]
        ensemble_prediction = fraud_results['ensemble_predictions'][0]
        
        # Get individual model scores
        model_scores = {name: scores[0] for name, scores in fraud_results['individual_scores'].items()}
        
        report = f"""
        FRAUD DETECTION REPORT
        ======================
        Transaction ID: {transaction_id}
        Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        TRANSACTION DETAILS:
        - Amount: ${transaction_data['amount'].iloc[0]:.2f}
        - Merchant Category: {transaction_data['merchant_category'].iloc[0]}
        - Transaction Type: {transaction_data['transaction_type'].iloc[0]}
        - Hour: {transaction_data['hour'].iloc[0]:02d}:00
        - Country: {transaction_data['country'].iloc[0]}
        - Customer Age: {transaction_data['customer_age'].iloc[0]:.0f} years
        
        FRAUD ANALYSIS:
        - Ensemble Fraud Score: {ensemble_score:.3f}
        - Fraud Prediction: {'FRAUDULENT' if ensemble_prediction == 1 else 'LEGITIMATE'}
        - Risk Level: {'HIGH' if ensemble_score > 0.7 else 'MEDIUM' if ensemble_score > 0.3 else 'LOW'}
        
        INDIVIDUAL MODEL SCORES:
        """
        
        for model_name, score in model_scores.items():
            report += f"- {model_name}: {score:.3f}\n"
        
        report += f"""
        RECOMMENDATION:
        """
        
        if ensemble_prediction == 1:
            if ensemble_score > 0.8:
                report += "- IMMEDIATE ACTION: Block transaction and flag account for review\n"
                report += "- CONTACT: Notify customer immediately\n"
                report += "- INVESTIGATE: Review recent transaction history\n"
            elif ensemble_score > 0.6:
                report += "- REVIEW REQUIRED: Hold transaction for manual review\n"
                report += "- VERIFY: Contact customer to confirm transaction\n"
                report += "- MONITOR: Flag account for increased monitoring\n"
            else:
                report += "- CAUTION: Transaction shows suspicious patterns\n"
                report += "- VERIFY: Consider additional verification\n"
                report += "- MONITOR: Increase monitoring for this account\n"
        else:
            if ensemble_score < 0.2:
                report += "- APPROVE: Transaction appears legitimate\n"
                report += "- NORMAL: Continue with standard processing\n"
            else:
                report += "- APPROVE: Transaction approved with caution\n"
                report += "- MONITOR: Consider increased monitoring\n"
        
        return report


def main():
    """
    Main function to demonstrate fraud detection system
    """
    print("Financial Fraud Detection System")
    print("=" * 50)
    
    # Initialize system
    fraud_system = FraudDetectionSystem()
    
    # Generate synthetic data
    print("Generating synthetic transaction data...")
    data = fraud_system.generate_synthetic_fraud_data(n_samples=50000)
    print(f"Generated {len(data)} transactions with {data['is_fraud'].sum()} fraudulent transactions")
    
    # Analyze fraud patterns
    fraud_system.analyze_fraud_patterns(data)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y = fraud_system.preprocess_data(data)
    print(f"Preprocessed {len(X)} samples with {len(X.columns)} features")
    
    # Train models
    print("\nTraining fraud detection models...")
    X_test, y_test = fraud_system.train_fraud_models(X, y)
    
    # Test fraud detection
    print("\nTesting fraud detection on new transactions...")
    sample_transactions = data.iloc[:5]
    fraud_results = fraud_system.detect_fraud(sample_transactions)
    
    # Visualize results
    print("\nGenerating visualizations...")
    fraud_system.visualize_fraud_analysis(data, fraud_results)
    
    # Generate sample report
    print("\nGenerating sample fraud report...")
    sample_transaction = data.iloc[:1]
    report = fraud_system.generate_fraud_report(1, sample_transaction, fraud_results)
    print(report)
    
    print("\nFraud detection system demonstration completed!")


if __name__ == "__main__":
    main() 