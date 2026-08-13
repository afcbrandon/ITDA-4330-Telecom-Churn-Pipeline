from sklearn.preprocessing import StandardScaler

from src.cleaner import clean_data
from src.features import engineer_features
from src.ingestion import load_and_merge_data


def run_pipeline():
    print("Starting Data Pipeline...")
    df = load_and_merge_data(
        "telecom_customer_churn.csv", "telecom_zipcode_population.csv"
    )
    df_clean = clean_data(df)
    df_engineered = engineer_features(df_clean)

    # Scikit-Learn Scaling Pipeline
    num_features = [
        "Tenure in Months",
        "Monthly Charge",
        "City_Target_Encoded",
        "Monthly_To_Tenure_Ratio",
    ]
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_engineered[num_features])

    scaled_cols = [f"{col}_Scaled" for col in num_features]
    df_engineered[scaled_cols] = scaled_array

    df_engineered.to_csv("output/telecom_churn_engineered.csv", index=False)
    print("Pipeline execution complete! Output saved to telecom_churn_engineered.csv")


if __name__ == "__main__":
    run_pipeline()
