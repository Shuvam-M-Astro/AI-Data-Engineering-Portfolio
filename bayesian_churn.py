import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load and preprocess data
# Download dataset from: https://www.kaggle.com/blastchar/telco-customer-churn
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if col not in ['customerID', 'Churn']:
        df[col] = LabelEncoder().fit_transform(df[col])

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Features and target
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
X = df[features]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Bayesian Logistic Regression
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=1, shape=X_train_scaled.shape[1])
    
    # Linear combination
    logits = alpha + pm.math.dot(X_train_scaled, betas)
    
    # Likelihood
    y_obs = pm.Bernoulli('y_obs', logit_p=logits, observed=y_train)
    
    # Inference
    trace = pm.sample(2000, tune=1000, target_accept=0.95, cores=2, random_seed=42)
    az.plot_trace(trace)
    az.summary(trace, hdi_prob=0.95).to_csv('churn_bayesian_summary.csv')

# 3. Posterior Predictive Checks
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=['y_obs'], random_seed=42)
    
y_pred = ppc['y_obs'].mean(axis=0)
print("Train Accuracy:", np.mean((y_pred > 0.5) == y_train))

# Predict on test set
with model:
    post_pred = pm.sample_posterior_predictive(trace, var_names=['alpha', 'betas'], random_seed=42)
    alpha_post = post_pred['alpha'].mean()
    betas_post = post_pred['betas'].mean(axis=0)
    logits_test = alpha_post + np.dot(X_test_scaled, betas_post)
    y_test_pred = 1 / (1 + np.exp(-logits_test))

print("Test Accuracy:", np.mean((y_test_pred > 0.5) == y_test)) 