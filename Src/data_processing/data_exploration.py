import pandas as pd

def explore_dataset():
    
    df = pd.read_csv('D:\MLOps_Churn_Prediction\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print("Dataset Shape:", df.shape)
    print(f"Dataset Columns: {list(df.columns)}")
    print("missing values:\n", df.isnull().sum())
    print("target variable distribution:\n", df['Churn'].value_counts())
    print("data types:\n", df.dtypes)
    
    df.describe().to_csv('D:\MLOps_Churn_Prediction\Data\data_summary2.csv')
    
    return df

if __name__ == "__main__":
    df= explore_dataset()
    print("Data exploration completed.-check dataset_description.csv for details.")
    