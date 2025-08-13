import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import mlflow

def preprocess_data(df):
    """Clean and preprocess the Telco churn dataset"""
    
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity']
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # Encode target variable
    df['Churn'] = le.fit_transform(df['Churn'])
    
    return df

def create_features(df):
    """Feature engineering"""
    # Create new features
    df['tenure_years'] = df['tenure'] / 12
    df['monthly_charges_per_service'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    return df

if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv('D:\MLOps_Churn_Prediction\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Process data
    df_processed = preprocess_data(df)
    df_processed = create_features(df_processed)
    
    # Save processed data
    df_processed.to_csv('D:\MLOps_Churn_Prediction\Data\processed_data.csv', index=False)
    print("Data preprocessing completed!")