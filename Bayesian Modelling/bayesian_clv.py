import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load and preprocess data
# Download dataset from: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
df = pd.read_excel('Online Retail.xlsx')
df = df[df['CustomerID'].notnull()]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Calculate CLV per customer
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
clv = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
clv.columns = ['CustomerID', 'CLV']

# Feature engineering: Recency, Frequency, Monetary
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Merge with CLV
data = pd.merge(rfm, clv, on='CustomerID')

# Prepare features
X = data[['Recency', 'Frequency', 'Monetary']]
y = data['CLV']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Bayesian Regression Model
with pm.Model() as model:
    # Priors for coefficients
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=1, shape=X_train_scaled.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Expected value
    mu = alpha + pm.math.dot(X_train_scaled, betas)
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
    
    # Inference
    trace = pm.sample(2000, tune=1000, target_accept=0.95, cores=2, random_seed=42)
    az.plot_trace(trace)
    az.summary(trace, hdi_prob=0.95).to_csv('clv_bayesian_summary.csv')

# 3. Posterior Predictive Checks
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=['y_obs'], random_seed=42)
    
y_pred = ppc['y_obs'].mean(axis=0)
print("Train RMSE:", np.sqrt(np.mean((y_pred - y_train) ** 2)))

# Predict on test set
with model:
    post_pred = pm.sample_posterior_predictive(trace, var_names=['alpha', 'betas', 'sigma'], random_seed=42)
    
    alpha_post = post_pred['alpha'].mean()
    betas_post = post_pred['betas'].mean(axis=0)
    y_test_pred = alpha_post + np.dot(X_test_scaled, betas_post)

print("Test RMSE:", np.sqrt(np.mean((y_test_pred - y_test) ** 2))) 