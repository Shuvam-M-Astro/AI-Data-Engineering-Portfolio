import os
# Disable MKL threading to avoid DLL issues
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Starting Alternative Bayesian Churn Analysis...")

# 1. Load and preprocess data
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please download the Telco Customer Churn dataset from Kaggle.")
    print("Place the file 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in this directory.")
    exit()

# Clean the data
print("Cleaning dataset...")
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Handle NaN values in Churn column
if 'Churn' in df.columns:
    # Remove rows where Churn is NaN
    df = df.dropna(subset=['Churn'])
    print(f"Removed {len(df)} rows with valid Churn values")
else:
    print("Error: 'Churn' column not found in dataset")
    exit()

# Convert Churn to numeric if it's not already
if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
elif df['Churn'].dtype == 'float64':
    # If it's already numeric, convert to int
    df['Churn'] = df['Churn'].astype(int)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if col not in ['customerID', 'Churn']:
        df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
X = df[features]
y = df['Churn']

print(f"Dataset shape: {df.shape}")
print(f"Features used: {features}")
print(f"Churn rate: {y.mean():.2%}")
print(f"Target variable info:")
print(f"- Data type: {y.dtype}")
print(f"- Unique values: {y.unique()}")
print(f"- NaN values: {y.isna().sum()}")

# Check for any remaining NaN values in features
print(f"\nChecking for NaN values in features:")
for col in features:
    nan_count = X[col].isna().sum()
    if nan_count > 0:
        print(f"- {col}: {nan_count} NaN values")
        # Fill NaN with median for numeric columns
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
            print(f"  Filled NaN values with median for {col}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 2. Frequentist Logistic Regression (baseline)
print("\nRunning Frequentist Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

print(f"Frequentist Model Performance:")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred_lr):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred_lr):.4f}")

# 3. Bayesian Analysis using Bootstrap Sampling
print("\nRunning Bayesian Analysis using Bootstrap Sampling...")

def bootstrap_logistic_regression(X, y, n_bootstrap=1000):
    """Perform bootstrap sampling to get Bayesian-like parameter estimates"""
    # Convert to numpy arrays to avoid pandas indexing issues
    X_np = np.array(X)
    y_np = np.array(y)
    
    n_samples, n_features = X_np.shape
    bootstrap_coeffs = []
    bootstrap_intercepts = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_np[indices]
        y_boot = y_np[indices]
        
        # Fit logistic regression
        try:
            model = LogisticRegression(random_state=i, max_iter=1000)
            model.fit(X_boot, y_boot)
            bootstrap_coeffs.append(model.coef_[0])
            bootstrap_intercepts.append(model.intercept_[0])
        except:
            continue
    
    return np.array(bootstrap_coeffs), np.array(bootstrap_intercepts)

# Run bootstrap sampling
bootstrap_coeffs, bootstrap_intercepts = bootstrap_logistic_regression(X_train_scaled, y_train, n_bootstrap=1000)

print(f"Bootstrap samples collected: {len(bootstrap_coeffs)}")

# 4. Calculate Bayesian-like statistics
coeff_means = np.mean(bootstrap_coeffs, axis=0)
coeff_stds = np.std(bootstrap_coeffs, axis=0)
intercept_mean = np.mean(bootstrap_intercepts)
intercept_std = np.std(bootstrap_intercepts)

print(f"\nBayesian-like Model Results (Bootstrap):")
print(f"Intercept: {intercept_mean:.4f} ± {intercept_std:.4f}")
for i, (mean, std) in enumerate(zip(coeff_means, coeff_stds)):
    print(f"Beta_{i} ({features[i]}): {mean:.4f} ± {std:.4f}")

# 5. Calculate credible intervals (95%)
coeff_ci_lower = np.percentile(bootstrap_coeffs, 2.5, axis=0)
coeff_ci_upper = np.percentile(bootstrap_coeffs, 97.5, axis=0)
intercept_ci_lower = np.percentile(bootstrap_intercepts, 2.5)
intercept_ci_upper = np.percentile(bootstrap_intercepts, 97.5)

print(f"\n95% Credible Intervals:")
print(f"Intercept: [{intercept_ci_lower:.4f}, {intercept_ci_upper:.4f}]")
for i, (lower, upper) in enumerate(zip(coeff_ci_lower, coeff_ci_upper)):
    print(f"Beta_{i} ({features[i]}): [{lower:.4f}, {upper:.4f}]")

# 6. Posterior Predictive Checks
print("\nPerforming posterior predictive checks...")

# Use bootstrap samples to make predictions
y_train_pred_bayes = []
y_test_pred_bayes = []

# Convert to numpy arrays for consistent indexing
X_train_scaled_np = np.array(X_train_scaled)
X_test_scaled_np = np.array(X_test_scaled)

for i in range(min(100, len(bootstrap_coeffs))):  # Use subset for efficiency
    # Predict on training set
    logits_train = bootstrap_intercepts[i] + np.dot(X_train_scaled_np, bootstrap_coeffs[i])
    probs_train = 1 / (1 + np.exp(-logits_train))
    y_train_pred_bayes.append(probs_train)
    
    # Predict on test set
    logits_test = bootstrap_intercepts[i] + np.dot(X_test_scaled_np, bootstrap_coeffs[i])
    probs_test = 1 / (1 + np.exp(-logits_test))
    y_test_pred_bayes.append(probs_test)

# Calculate mean predictions
y_train_pred_mean = np.mean(y_train_pred_bayes, axis=0)
y_test_pred_mean = np.mean(y_test_pred_bayes, axis=0)

# Calculate accuracy
train_accuracy_bayes = np.mean((y_train_pred_mean > 0.5) == y_train)
test_accuracy_bayes = np.mean((y_test_pred_mean > 0.5) == y_test)

print(f"Bayesian Model Performance:")
print(f"Train Accuracy: {train_accuracy_bayes:.4f}")
print(f"Test Accuracy: {test_accuracy_bayes:.4f}")

# 7. Save results
results_df = pd.DataFrame({
    'Parameter': ['intercept'] + features,
    'Mean': [intercept_mean] + list(coeff_means),
    'Std': [intercept_std] + list(coeff_stds),
    'CI_Lower': [intercept_ci_lower] + list(coeff_ci_lower),
    'CI_Upper': [intercept_ci_upper] + list(coeff_ci_upper)
})

results_df.to_csv('churn_bayesian_bootstrap_summary.csv', index=False)
print(f"\nResults saved to 'churn_bayesian_bootstrap_summary.csv'")

# 8. Visualization
plt.figure(figsize=(15, 10))

# Plot coefficient distributions
for i in range(len(features)):
    plt.subplot(2, 3, i+1)
    plt.hist(bootstrap_coeffs[:, i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(coeff_means[i], color='red', linestyle='--', label=f'Mean: {coeff_means[i]:.3f}')
    plt.axvline(coeff_ci_lower[i], color='orange', linestyle=':', label=f'95% CI: [{coeff_ci_lower[i]:.3f}, {coeff_ci_upper[i]:.3f}]')
    plt.axvline(coeff_ci_upper[i], color='orange', linestyle=':')
    plt.title(f'{features[i]} Coefficient Distribution')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.savefig('churn_bayesian_bootstrap_plots.png', dpi=300, bbox_inches='tight')
print("Plots saved to 'churn_bayesian_bootstrap_plots.png'")

# 9. Feature importance analysis
feature_importance = np.abs(coeff_means)
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Mean_Effect': coeff_means,
    'Std_Effect': coeff_stds,
    'Abs_Effect': feature_importance
}).sort_values('Abs_Effect', ascending=False)

print(f"\nFeature Importance (by absolute effect size):")
for _, row in feature_importance_df.iterrows():
    print(f"{row['Feature']}: {row['Mean_Effect']:.4f} ± {row['Std_Effect']:.4f}")

# 10. Model comparison
print(f"\nModel Comparison:")
print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 44)
print(f"{'Frequentist':<20} {accuracy_score(y_train, y_train_pred_lr):<12.4f} {accuracy_score(y_test, y_test_pred_lr):<12.4f}")
print(f"{'Bayesian (Bootstrap)':<20} {train_accuracy_bayes:<12.4f} {test_accuracy_bayes:<12.4f}")

print(f"\nAnalysis completed successfully!")
print(f"Key findings:")
print(f"- Dataset has {len(df)} customers with {y.mean():.1%} churn rate")
print(f"- Bayesian analysis used {len(bootstrap_coeffs)} bootstrap samples")
print(f"- Most important feature: {feature_importance_df.iloc[0]['Feature']}")
print(f"- Model performance is similar between frequentist and Bayesian approaches") 