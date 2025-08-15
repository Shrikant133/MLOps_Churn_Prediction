# src/data_processing/data_cleaning.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def clean_telco_data(df):
    """Clean the Telco churn dataset"""
    print(" Starting data cleaning...")
    df_clean = df.copy()
    
    # 1. Fix TotalCharges - convert to numeric (has empty strings)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill missing values with median
    missing_charges = df_clean['TotalCharges'].isnull().sum()
    if missing_charges > 0:
        median_charges = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'].fillna(median_charges, inplace=True)
        print(f"   Fixed {missing_charges} missing values in TotalCharges")
    
    # 2. Remove customerID (not needed for modeling)
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
        print("   Removed customerID column")
    
    # 3. Remove duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"   Removed {duplicates} duplicate rows")
    
    print(f" Data cleaning completed. Shape: {df_clean.shape}")
    return df_clean

def encode_categorical_data(df):
    """Convert categorical variables to numerical format"""
    print(" Starting categorical encoding...")
    df_encoded = df.copy()
    
    # Binary Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
    # Gender
    if 'gender' in df_encoded.columns:
        df_encoded['gender'] = df_encoded['gender'].map({'Male': 1, 'Female': 0})
    
    # Target variable (Churn)
    if 'Churn' in df_encoded.columns:
        df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
    
    # MultipleLines (3 categories)
    if 'MultipleLines' in df_encoded.columns:
        df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({
            'Yes': 2, 'No': 1, 'No phone service': 0
        })
    
    # Internet service related columns
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({
                'Yes': 2, 'No': 1, 'No internet service': 0
            })
    
    # InternetService
    if 'InternetService' in df_encoded.columns:
        df_encoded['InternetService'] = df_encoded['InternetService'].map({
            'Fiber optic': 2, 'DSL': 1, 'No': 0
        })
    
    # Contract
    if 'Contract' in df_encoded.columns:
        df_encoded['Contract'] = df_encoded['Contract'].map({
            'Month-to-month': 0, 'One year': 1, 'Two year': 2
        })
    
    # PaymentMethod (use label encoding)
    if 'PaymentMethod' in df_encoded.columns:
        le_payment = LabelEncoder()
        df_encoded['PaymentMethod'] = le_payment.fit_transform(df_encoded['PaymentMethod'])
    
    print(" Categorical encoding completed")
    return df_encoded

def create_new_features(df):
    """Create advanced features for 90%+ model performance"""
    print("⚒️ Creating advanced features for better performance...")
    df_features = df.copy()
    
    # 1. Tenure-based features
    df_features['tenure_years'] = df_features['tenure'] / 12
    df_features['is_new_customer'] = (df_features['tenure'] <= 6).astype(int)  # First 6 months
    df_features['is_loyal_customer'] = (df_features['tenure'] >= 48).astype(int)  # 4+ years
    
    # 2. Financial features
    df_features['monthly_charges_per_tenure'] = df_features['MonthlyCharges'] / (df_features['tenure'] + 1)
    df_features['total_to_monthly_ratio'] = df_features['TotalCharges'] / df_features['MonthlyCharges']
    df_features['avg_monthly_charges'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)
    
    # 3. Service bundle features
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_features['service_usage_score'] = 0
    for col in service_cols:
        if col in df_features.columns:
            df_features['service_usage_score'] += (df_features[col] == 2).astype(int)
    
    df_features['has_premium_services'] = (df_features['service_usage_score'] >= 3).astype(int)
    df_features['has_streaming_services'] = ((df_features['StreamingTV'] == 2) | (df_features['StreamingMovies'] == 2)).astype(int)
    
    # 4. Contract and payment risk features
    df_features['is_month_to_month'] = (df_features['Contract'] == 0).astype(int)  # High churn risk
    df_features['has_paperless_billing'] = df_features['PaperlessBilling']
    df_features['electronic_payment'] = (df_features['PaymentMethod'].isin([0, 1])).astype(int)  # Higher churn risk
    
    # 5. High-value customer segments
    monthly_75th = df_features['MonthlyCharges'].quantile(0.75)
    total_75th = df_features['TotalCharges'].quantile(0.75)
    
    df_features['high_value_customer'] = (
        (df_features['MonthlyCharges'] > monthly_75th) & 
        (df_features['TotalCharges'] > total_75th)
    ).astype(int)
    
    # 6. Churn risk indicators
    df_features['churn_risk_score'] = (
        df_features['is_month_to_month'] * 2 +  # Month-to-month contract
        df_features['is_new_customer'] * 2 +    # New customers churn more
        (df_features['service_usage_score'] == 0) * 1 +  # No additional services
        df_features['electronic_payment'] * 1   # Electronic payment methods
    )
    
    # 7. Internet service quality features
    df_features['has_fiber_optic'] = (df_features['InternetService'] == 2).astype(int)
    df_features['internet_service_quality'] = df_features['InternetService']  # 0=No, 1=DSL, 2=Fiber
    
    # 8. Customer lifecycle features
    df_features['charges_per_service'] = df_features['MonthlyCharges'] / (df_features['service_usage_score'] + 1)
    df_features['retention_potential'] = (
        (df_features['tenure'] > 12) & 
        (df_features['service_usage_score'] >= 2) & 
        (df_features['Contract'] > 0)
    ).astype(int)
    
    print(" Advanced feature engineering completed - 15+ new features created")
    print(f"   Total features: {len(df_features.columns)}")
    return df_features

def process_telco_data(input_file, output_file):
    """Complete data processing pipeline"""
    print(" Starting Telco data processing...")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f" Loaded data: {df.shape}")
    
    # Clean data
    df_clean = clean_telco_data(df)
    
    # Encode categorical variables
    df_encoded = encode_categorical_data(df_clean)
    
    # Create new features
    df_processed = create_new_features(df_encoded)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    
    print(f" Processed data saved: {output_file}")
    print(f" Final shape: {df_processed.shape}")
    print(" Data processing completed!")
    
    return df_processed

if __name__ == "__main__":
    # Process the data
    input_path = "D:\MLOps_Churn_Prediction\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path = "D:\MLOps_Churn_Prediction\Data\data_processed.csv"
    
    processed_df = process_telco_data(input_path, output_path)