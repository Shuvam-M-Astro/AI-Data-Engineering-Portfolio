import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

print("Generating sample Telco Customer Churn dataset...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data that mimics the Telco Customer Churn dataset
n_samples = 7043  # Same size as original dataset

# Create synthetic features that mimic real churn data
tenure = np.random.exponential(scale=30, size=n_samples)  # Customer tenure (months)
tenure = np.clip(tenure, 1, 72)  # Limit to reasonable range

monthly_charges = np.random.normal(65, 25, n_samples)  # Monthly charges
monthly_charges = np.clip(monthly_charges, 18, 120)  # Limit to reasonable range

total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
total_charges = np.clip(total_charges, 0, 10000)  # Limit to reasonable range

# Contract types (0=Month-to-month, 1=One year, 2=Two year)
contract = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])

# Internet service (0=No, 1=DSL, 2=Fiber optic)
internet_service = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.4, 0.4])

# Payment method (0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card)
payment_method = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.2, 0.25, 0.25])

# Create customer IDs
customer_ids = [f'7590-VHVEG_{i:04d}' for i in range(n_samples)]

# Create synthetic churn based on realistic patterns
# Higher churn for month-to-month contracts, higher charges, etc.
churn_prob = (
    0.1 +  # Base churn rate
    0.3 * (contract == 0) +  # Higher churn for month-to-month
    0.2 * (monthly_charges > 70) +  # Higher churn for expensive plans
    0.15 * (tenure < 12) +  # Higher churn for new customers
    0.1 * (payment_method == 0)  # Higher churn for electronic check
)
churn_prob = np.clip(churn_prob, 0, 1)
churn = np.random.binomial(1, churn_prob, size=n_samples)

# Create the dataset
data = {
    'customerID': customer_ids,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'InternetService': internet_service,
    'PaymentMethod': payment_method,
    'Churn': churn
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Add some additional realistic features
df['gender'] = np.random.choice(['Male', 'Female'], size=n_samples)
df['SeniorCitizen'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
df['Partner'] = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.5, 0.5])
df['Dependents'] = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
df['PhoneService'] = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.9, 0.1])
df['MultipleLines'] = np.random.choice(['Yes', 'No', 'No phone service'], size=n_samples, p=[0.4, 0.4, 0.2])
df['OnlineSecurity'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.3, 0.5, 0.2])
df['OnlineBackup'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.3, 0.5, 0.2])
df['DeviceProtection'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.3, 0.5, 0.2])
df['TechSupport'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.2, 0.6, 0.2])
df['StreamingTV'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.4, 0.4, 0.2])
df['StreamingMovies'] = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples, p=[0.4, 0.4, 0.2])
df['PaperlessBilling'] = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.6, 0.4])

# Save the dataset
df.to_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)

print(f"Sample dataset created with {len(df)} customers")
print(f"Churn rate: {df['Churn'].mean():.2%}")
print(f"Dataset saved as 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")

# Display some statistics
print(f"\nDataset Statistics:")
print(f"Tenure (months): {df['tenure'].mean():.1f} ± {df['tenure'].std():.1f}")
print(f"Monthly Charges: ${df['MonthlyCharges'].mean():.1f} ± ${df['MonthlyCharges'].std():.1f}")
print(f"Total Charges: ${df['TotalCharges'].mean():.1f} ± ${df['TotalCharges'].std():.1f}")
print(f"Contract distribution: Month-to-month: {(df['Contract']==0).mean():.1%}, One year: {(df['Contract']==1).mean():.1%}, Two year: {(df['Contract']==2).mean():.1%}")
print(f"Internet Service: No: {(df['InternetService']==0).mean():.1%}, DSL: {(df['InternetService']==1).mean():.1%}, Fiber: {(df['InternetService']==2).mean():.1%}")

print(f"\nYou can now run the Bayesian analysis script!") 