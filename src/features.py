import pandas as pd


def engineer_features(df):
    # Categorical Encoding
    city_churn_map = df.groupby("City")["Churn_Binary"].mean()
    df["City_Target_Encoded"] = df["City"].map(city_churn_map)

    payment_freq = df["Payment Method"].value_counts(normalize=True)
    df["PaymentMethod_Freq_Encoded"] = df["Payment Method"].map(payment_freq)

    # Feature Creation & Interactions
    df["Monthly_To_Tenure_Ratio"] = df["Monthly Charge"] / (df["Tenure in Months"] + 1)
    df["Revenue_Per_Capita_Index"] = df["Total Revenue"] / (df["Population"] + 1)

    # Time Series & Tenure Metrics
    df["Tenure_Years"] = df["Tenure in Months"] / 12.0
    df["Tenure_Cohort"] = pd.cut(
        df["Tenure in Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1 Yr", "1-2 Yrs", "2-4 Yrs", "4+ Yrs"],
    )

    return df
