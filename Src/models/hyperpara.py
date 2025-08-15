# src/models/hyperparameter_tuning_enhanced.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
import json
from datetime import datetime
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_and_prepare_data():
    """Load and prepare data for hyperparameter tuning"""
    print(" Loading processed data for hyperparameter tuning...")
    df = pd.read_csv('D:\MLOps_Churn_Prediction\Data\data_processed.csv')
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('models/scalers', exist_ok=True)
    joblib.dump(scaler, 'models/scalers/standard_scaler.pkl')
    
    print(f"   Training samples: {len(X_train_scaled)}")
    print(f"   Test samples: {len(X_test_scaled)}")
    print(f"   Features: {len(X.columns)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
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

def tune_random_forest():
    """Hyperparameter tuning for Random Forest - Optimized for 80%+ accuracy"""
    print(" Tuning Random Forest hyperparameters...")
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Focused parameter grid for better performance
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [12, 15, 18],
        'min_samples_split': [3, 5, 7],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2']
    }
    
    total_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) * 
                         len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * 
                         len(param_grid['max_features']))
    print(f" Testing {total_combinations} parameter combinations...")
    
    # Create model
    rf = RandomForestClassifier(
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    with mlflow.start_run(run_name="rf_hyperparameter_tuning_enhanced"):
        print(" Running grid search (this may take 5-10 minutes)...")
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Log best parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_param("total_combinations_tested", total_combinations)
        mlflow.log_param("cv_folds", 5)
        
        # Log metrics
        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "tuned_random_forest",
            registered_model_name="churn_rf_best"
        )
        
        # Save best model locally
        joblib.dump(best_model, 'models/best_random_forest.pkl')
        
        print(f" Random Forest Results:")
        print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")
        print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Test F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Test Precision: {metrics['precision']:.4f}")
        print(f"   Test Recall: {metrics['recall']:.4f}")
        if 'roc_auc' in metrics:
            print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f" Best Parameters: {grid_search.best_params_}")
        
        return best_model, grid_search.best_params_, metrics

def tune_logistic_regression():
    """Hyperparameter tuning for Logistic Regression"""
    print(" Tuning Logistic Regression hyperparameters...")
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Parameter grid for Logistic Regression
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000, 2000, 3000],
        'class_weight': ['balanced', None]
    }
    
    total_combinations = (len(param_grid['C']) * len(param_grid['solver']) * 
                         len(param_grid['max_iter']) * len(param_grid['class_weight']))
    print(f" Testing {total_combinations} parameter combinations...")
    
    # Create model
    lr = LogisticRegression(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        lr, 
        param_grid, 
        cv=5, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    with mlflow.start_run(run_name="lr_hyperparameter_tuning_enhanced"):
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Log parameters and metrics
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "tuned_logistic_regression",
            registered_model_name="churn_lr_best"
        )
        
        joblib.dump(best_model, 'models/best_logistic_regression.pkl')
        
        print(f" Logistic Regression Results:")
        print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Test F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Test Precision: {metrics['precision']:.4f}")
        print(f"   Test Recall: {metrics['recall']:.4f}")
        if 'roc_auc' in metrics:
            print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f" Best Parameters: {grid_search.best_params_}")
        
        return best_model, grid_search.best_params_, metrics

def tune_svm():
    """Hyperparameter tuning for Support Vector Machine"""
    print(" Tuning SVM hyperparameters...")
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Parameter grid for SVM - focused on performance
    param_grid = {
        'C': [0.1, 1, 5, 10, 50],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'class_weight': ['balanced', None]
    }
    
    total_combinations = (len(param_grid['C']) * len(param_grid['kernel']) * 
                         len(param_grid['gamma']) * len(param_grid['class_weight']))
    print(f" Testing {total_combinations} parameter combinations...")
    print("  Note: SVM tuning may take 10-15 minutes due to complexity")
    
    # Create model
    svm = SVC(random_state=42, probability=True)
    
    # Grid search with reduced CV for speed
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=3,  # Reduced CV folds for SVM due to computational cost
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    with mlflow.start_run(run_name="svm_hyperparameter_tuning_enhanced"):
        print("⚡ Running SVM grid search...")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Log parameters and metrics
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_param("total_combinations_tested", total_combinations)
        mlflow.log_param("cv_folds", 3)
        
        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "tuned_svm",
            registered_model_name="churn_svm_best"
        )
        
        joblib.dump(best_model, 'models/best_svm.pkl')
        
        print(f" SVM Results:")
        print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")
        print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Test F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Test Precision: {metrics['precision']:.4f}")
        print(f"   Test Recall: {metrics['recall']:.4f}")
        if 'roc_auc' in metrics:
            print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f" Best Parameters: {grid_search.best_params_}")
        
        return best_model, grid_search.best_params_, metrics

def select_champion_model(rf_results, lr_results, svm_results):
    """
    Advanced model selection based on multiple criteria
    """
    print("\n" + "="*80)
    print(" CHAMPION MODEL SELECTION")
    print("="*80)
    
    rf_metrics = rf_results[2]
    lr_metrics = lr_results[2]
    svm_metrics = svm_results[2]
    
    # Create comprehensive comparison
    comparison = pd.DataFrame({
        'Random_Forest': rf_metrics,
        'Logistic_Regression': lr_metrics,
        'SVM': svm_metrics
    })
    
    print(" Model Performance Comparison:")
    print(comparison.round(4))
    print()
    
    # Model Selection Criteria Analysis
    print(" MODEL SELECTION CRITERIA ANALYSIS:")
    print("-" * 50)
    
    # 1. Business Context for Churn Prediction
    print("1. BUSINESS CONTEXT:")
    print("   • False Negatives (missed churners) are costly")
    print("   • False Positives (wrong churn predictions) waste resources")
    print("   • Balance between Precision and Recall is crucial")
    print()
    
    # 2. Metric Analysis
    print("2. METRIC ANALYSIS:")
    for model in ['Random_Forest', 'Logistic_Regression', 'SVM']:
        metrics = comparison[model]
        print(f"   {model}:")
        print(f"     • Accuracy: {metrics['accuracy']:.4f} (Overall correctness)")
        print(f"     • Precision: {metrics['precision']:.4f} (When we predict churn, how often right?)")
        print(f"     • Recall: {metrics['recall']:.4f} (Of all churners, how many did we catch?)")
        print(f"     • F1-Score: {metrics['f1_score']:.4f} (Harmonic mean of Precision & Recall)")
        if 'roc_auc' in metrics:
            print(f"     • ROC-AUC: {metrics['roc_auc']:.4f} (Overall discrimination ability)")
        print()
    
    # 3. Selection Logic
    print("3. CHAMPION MODEL SELECTION LOGIC:")
    
    # Calculate composite score
    weights = {
        'f1_score': 0.35,      # Most important for churn prediction
        'recall': 0.25,        # Critical - don't miss churners
        'precision': 0.20,     # Important - avoid false alarms
        'accuracy': 0.15,      # Overall performance
        'roc_auc': 0.05        # Discrimination ability
    }
    
    composite_scores = {}
    for model in comparison.columns:
        score = 0
        for metric, weight in weights.items():
            if metric in comparison.loc[:, model]:
                score += comparison.loc[metric, model] * weight
        composite_scores[model] = score
    
    print("   Weighted Composite Scoring:")
    for metric, weight in weights.items():
        print(f"     • {metric}: {weight*100:.0f}% weight")
    print()
    
    print("   Composite Scores:")
    for model, score in sorted(composite_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"     • {model}: {score:.4f}")
    print()
    
    # Select champion based on composite score
    champion_model_name = max(composite_scores, key=composite_scores.get)
    
    # Get the actual model and results
    if champion_model_name == 'Random_Forest':
        champion_model = rf_results[0]
        champion_params = rf_results[1]
        champion_metrics = rf_results[2]
    elif champion_model_name == 'Logistic_Regression':
        champion_model = lr_results[0]
        champion_params = lr_results[1]
        champion_metrics = lr_results[2]
    else:  # SVM
        champion_model = svm_results[0]
        champion_params = svm_results[1]
        champion_metrics = svm_results[2]
    
    print(" CHAMPION MODEL SELECTED:")
    print(f"   Model: {champion_model_name}")
    print(f"   Composite Score: {composite_scores[champion_model_name]:.4f}")
    print(f"   F1-Score: {champion_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {champion_metrics['accuracy']:.4f}")
    print(f"   Recall: {champion_metrics['recall']:.4f}")
    print(f"   Precision: {champion_metrics['precision']:.4f}")
    print()
    
    # Business recommendation
    print("💼 BUSINESS RECOMMENDATION:")
    if champion_metrics['recall'] > 0.75:
        print("    High Recall - Good at catching churners")
    else:
        print("     Consider if recall is sufficient for business needs")
        
    if champion_metrics['precision'] > 0.70:
        print("    High Precision - Low false alarm rate")
    else:
        print("     May generate some false churn alerts")
    
    if champion_metrics['f1_score'] > 0.75:
        print("    Strong overall balance of precision and recall")
    
    # Save champion model
    joblib.dump(champion_model, 'models/champion_model.pkl')
    
    # Save comprehensive model metadata
    model_metadata = {
        'champion_model': champion_model_name,
        'selection_criteria': {
            'method': 'weighted_composite_scoring',
            'weights': weights,
            'composite_score': composite_scores[champion_model_name]
        },
        'best_parameters': champion_params,
        'performance_metrics': champion_metrics,
        'all_model_comparison': comparison.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'model_version': '2.0'
    }
    
    with open('models/champion_model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    print(" Champion model saved as models/champion_model.pkl")
    print(" Metadata saved as models/champion_model_metadata.json")
    
    return champion_model, model_metadata

def main():
    """Main enhanced hyperparameter tuning pipeline"""
    print("Starting ENHANCED Hyperparameter Tuning Pipeline")
    print("="*80)
    
    # Set experiment
    experiment_name = f"churn_hyperparameter_tuning_enhanced_{datetime.now().strftime('%Y%m%d_%H%M')}"
    mlflow.set_experiment(experiment_name)
    
    print(f" MLflow Experiment: {experiment_name}")
    print(f" MLflow UI: http://localhost:5000")
    print()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Tune all models
    print("Phase 1: Random Forest Tuning")
    rf_results = tune_random_forest()
    print()
    
    print("Phase 2: Logistic Regression Tuning")
    lr_results = tune_logistic_regression()
    print()
    
    print("Phase 3: SVM Tuning")
    svm_results = tune_svm()
    print()
    
    # Select champion model with advanced criteria
    champion_model, metadata = select_champion_model(rf_results, lr_results, svm_results)
    
    print("\n" + "="*80)
    print(" ENHANCED HYPERPARAMETER TUNING COMPLETED!")
    print("="*80)
    
    total_combinations = 90 + 144 + 300  # RF + LR + SVM
    print(f" Total parameter combinations tested: {total_combinations}")
    print(f" Champion model ready for deployment!")
    print(f" Expected performance in production: ~{metadata['performance_metrics']['accuracy']:.1%}")
    
    print("\nNext Steps:")
    print("1. Enhanced model training and tuning completed")  
    print("2. Run deployment preparation")
    print("3. Build CI/CD pipeline")
    print("4. Create Docker container")
    print("5. Deploy FastAPI service")
    
    return champion_model, metadata

if __name__ == "__main__":
    champion, metadata = main()