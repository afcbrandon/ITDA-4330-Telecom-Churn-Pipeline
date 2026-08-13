from sklearn.impute import SimpleImputer


def clean_data(df):
    # Remove out-of-range numerical anomalies
    df_clean = df[df["Monthly Charge"] >= 0].copy()

    # Handle missing data
    impute_zero_cols = ["Avg Monthly Long Distance Charges", "Avg Monthly GB Download"]
    df_clean[impute_zero_cols] = df_clean[impute_zero_cols].fillna(0)

    simple_imputer = SimpleImputer(strategy="mean")
    df_clean["Population"] = simple_imputer.fit_transform(df_clean[["Population"]])

    df_clean["Churn_Binary"] = (df_clean["Customer Status"] == "Churned").astype(int)
    return df_clean
