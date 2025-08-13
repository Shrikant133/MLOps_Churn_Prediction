import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os

# ================================
# Load processed churn dataset
# ================================
df = pd.read_csv(r"D:\MLOps_Churn_Prediction\Data\data_processed.csv")

# Set up MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_prediction_eda")

with mlflow.start_run(run_name="EDA_Visualization"):

    # Create folder for plots
    os.makedirs("eda_plots", exist_ok=True)

    # 1. Churn distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette="viridis")
    plt.title("Churn Distribution")
    plt.savefig("eda_plots/churn_distribution.png")
    mlflow.log_artifact("eda_plots/churn_distribution.png")
    plt.close()

    # 2. Churn vs Contract Type
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Contract', hue='Churn', data=df, palette="viridis")
    plt.title("Churn by Contract Type")
    plt.xticks(rotation=30)
    plt.savefig("eda_plots/churn_vs_contract.png")
    mlflow.log_artifact("eda_plots/churn_vs_contract.png")
    plt.close()

    # 3. Churn vs Monthly Charges
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df[df['Churn'] == 0]['MonthlyCharges'], shade=True, label="No Churn")
    sns.kdeplot(df[df['Churn'] == 1]['MonthlyCharges'], shade=True, label="Churn")
    plt.title("Monthly Charges Distribution by Churn")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("eda_plots/churn_vs_monthlycharges.png")
    mlflow.log_artifact("eda_plots/churn_vs_monthlycharges.png")
    plt.close()

    # 4. Churn vs Tenure
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df[df['Churn'] == 0]['tenure'], shade=True, label="No Churn")
    sns.kdeplot(df[df['Churn'] == 1]['tenure'], shade=True, label="Churn")
    plt.title("Tenure Distribution by Churn")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("eda_plots/churn_vs_tenure.png")
    mlflow.log_artifact("eda_plots/churn_vs_tenure.png")
    plt.close()

    # 5. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("eda_plots/correlation_heatmap.png")
    mlflow.log_artifact("eda_plots/correlation_heatmap.png")
    plt.close()

    mlflow.log_param("eda_plots_count", 5)

print("EDA plots saved and logged to MLflow.")
