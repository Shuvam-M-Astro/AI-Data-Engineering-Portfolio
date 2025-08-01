import pandas as pd
import numpy as np

print("Checking dataset structure and quality...")

try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nChurn column analysis:")
    if 'Churn' in df.columns:
        print(f"Churn unique values: {df['Churn'].unique()}")
        print(f"Churn value counts:")
        print(df['Churn'].value_counts())
        print(f"Churn data type: {df['Churn'].dtype}")
        print(f"Churn NaN count: {df['Churn'].isna().sum()}")
    else:
        print("'Churn' column not found!")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nSample of rows with potential issues:")
    # Check for rows with empty strings or unusual values
    for col in ['TotalCharges', 'Churn']:
        if col in df.columns:
            empty_mask = df[col] == ' '
            if empty_mask.any():
                print(f"Rows with empty '{col}': {empty_mask.sum()}")
                print(df[empty_mask].head())
    
except Exception as e:
    print(f"Error reading dataset: {e}") 