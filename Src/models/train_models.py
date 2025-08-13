import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from datetime import datetime

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_processed_data():
    """Load the processed data"""
    try:
        df = pd.read_csv('D:\MLOps_Churn_Prediction\Data\data_processed.csv')
        print(f"Processed data loaded: {df.shape}")
        return df
    except FileNotFoundError:
        print("Processed data not found. Please run data preprocessing first.")
        return None

def prepare_data_for_training(df, test_size=0.2, random_state=42):
    """Split and scale the data"""
    print("Preparing data for training...")
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    os.makedirs('models/scalers', exist_ok=True)
    joblib.dump(scaler, 'models/scalers/standard_scaler.pkl')
    
    print(f"   Training set: {X_train_scaled.shape}")
    print(f"   Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Train Logistic Regression model"""
    print("Training Logistic Regression...")
    
    with mlflow.start_run(run_name="logistic_regression"):
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", len(feature_names))
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=X_train[:5],
            registered_model_name="churn_logistic_regression"
        )
        
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return model, metrics

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train Random Forest model"""
    print("Training Random Forest...")
    
    with mlflow.start_run(run_name="random_forest"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", len(feature_names))
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance as artifact
        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)  # Clean up
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=X_train[:5],
            registered_model_name="churn_random_forest"
        )
        
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return model, metrics

def train_svm(X_train, X_test, y_train, y_test, feature_names):
    """Train Support Vector Machine model"""
    print("Training SVM...")
    
    with mlflow.start_run(run_name="support_vector_machine"):
        # Train model
        model = SVC(
            kernel='rbf',
            C=1.0,
            random_state=42,
            probability=True  # Enable probability predictions
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Log parameters
        mlflow.log_param("model_type", "SVC")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", len(feature_names))
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=X_train[:5],
            registered_model_name="churn_svm"
        )
        
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return model, metrics

def compare_models(model_results):
    """Compare all trained models"""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison_df = pd.DataFrame(model_results).T
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    
    print(comparison_df.round(4))
    
    best_model = comparison_df.index[0]
    best_f1 = comparison_df.loc[best_model, 'f1_score']
    
    print(f"\n Best Model: {best_model}")
    print(f"Best F1-Score: {best_f1:.4f}")
    
    return comparison_df

def main():
    """Main training pipeline"""
    print("Starting MLflow Model Training Pipeline")
    print("="*60)
    
    # Set experiment
    experiment_name = f"churn_prediction_{datetime.now().strftime('%Y%m%d')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow Experiment: {experiment_name}")
    print(f"MLflow UI: http://localhost:5000")
    
    # Load data
    df = load_processed_data()
    if df is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_training(df)
    
    # Train models
    model_results = {}
    
    # 1. Logistic Regression
    model_lr, metrics_lr = train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)
    model_results['logistic_regression'] = metrics_lr
    
    # 2. Random Forest
    model_rf, metrics_rf = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    model_results['random_forest'] = metrics_rf
    
    # 3. SVM
    model_svm, metrics_svm = train_svm(X_train, X_test, y_train, y_test, feature_names)
    model_results['svm'] = metrics_svm
    
    # Compare models
    comparison_df = compare_models(model_results)
    
    # Save comparison results
    comparison_df.to_csv('models/model_comparison.csv')
    print(f"\n Model comparison saved to models/model_comparison.csv")
    
    print("\n" + "="*60)
    print(" MODEL TRAINING COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("1. Check MLflow UI: http://localhost:5000")
    print("2. Review model comparison results")
    print("3. Select best model for deployment")
    
    return model_results

if __name__ == "__main__":
    results = main()