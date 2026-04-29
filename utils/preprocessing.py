import pandas as pd


# 🔹 1. Load dataset
def load_data():
    df = pd.read_csv("data/supermarket_dummy_data.csv")

    # convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    return df


# 🔹 2. Handle missing values
def handle_missing_values(df):
    df = df.fillna(0)
    return df


# 🔹 3. Add profit column
def add_profit(df):
    df["Profit"] = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity_Sold"]
    return df


# 🔹 4. Add time-based features
def add_time_features(df):
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    return df


# 🔹 5. Ensure Festival column exists
def ensure_festival_column(df):
    if "Festival" not in df.columns:
        df["Festival"] = 0
    return df


# 🔹 6. Full preprocessing pipeline
def preprocess_data():
    df = load_data()

    df = handle_missing_values(df)

    df = ensure_festival_column(df)

    df = add_profit(df)

    df = add_time_features(df)

    return df


# 🔹 7. Optional: Filter data by date range
def filter_by_date(df, start_date=None, end_date=None):
    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]
    return df


# 🔹 8. Optional: Get basic summary
def data_summary(df):
    summary = {
        "Total Rows": len(df),
        "Total Products": df["Product_Name"].nunique(),
        "Total Customers": df["Customer_ID"].nunique(),
        "Total Profit": df["Profit"].sum()
    }
    return summary


# 🔹 9. Test run
if __name__ == "__main__":
    df = preprocess_data()

    print("Data Loaded Successfully ✅")
    print(df.head())

    print("\nSummary:")
    print(data_summary(df))