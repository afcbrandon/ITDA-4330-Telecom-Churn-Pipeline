import pandas as pd


def load_and_merge_data(churn_path, pop_path):
    df_churn = pd.read_csv(churn_path)
    df_pop = pd.read_csv(pop_path)
    merged_df = pd.merge(df_churn, df_pop, on="Zip Code", how="left")
    return merged_df


if __name__ == "__main__":
    df = load_and_merge_data(
        "telecom_customer_churn.csv", "telecom_zipcode_population.csv"
    )
    print(f"Ingested & merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")
