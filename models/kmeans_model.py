import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 🔹 1. Prepare customer-level data
def prepare_customer_data(df):
    customer_df = df.groupby("Customer_ID").agg({
        "Quantity_Sold": "sum",
        "Profit": "sum"
    }).reset_index()

    return customer_df


# 🔹 2. Scale features (important for K-Means)
def scale_features(customer_df):
    scaler = StandardScaler()

    features = customer_df[["Quantity_Sold", "Profit"]]
    scaled_features = scaler.fit_transform(features)

    return scaled_features


# 🔹 3. Apply K-Means clustering
def apply_kmeans(customer_df, n_clusters=3):
    scaled_features = scale_features(customer_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_df["Cluster"] = kmeans.fit_predict(scaled_features)

    return customer_df


# 🔹 4. Full pipeline (main function)
def segment_customers(df):
    customer_df = prepare_customer_data(df)
    customer_df = apply_kmeans(customer_df)

    return customer_df


# 🔹 5. Cluster summary (for analysis)
def cluster_summary(df):
    customer_df = segment_customers(df)

    summary = customer_df.groupby("Cluster").agg({
        "Quantity_Sold": "mean",
        "Profit": "mean",
        "Customer_ID": "count"
    }).rename(columns={"Customer_ID": "Num_Customers"}).reset_index()

    return summary


# 🔹 6. Optional: Label clusters (for readability)
def label_clusters(df):
    customer_df = segment_customers(df)

    # get average profit per cluster
    profit_avg = customer_df.groupby("Cluster")["Profit"].mean()

    # sort clusters by profit
    sorted_clusters = profit_avg.sort_values().index.tolist()

    labels = {
        sorted_clusters[0]: "Low Value",
        sorted_clusters[1]: "Medium Value",
        sorted_clusters[2]: "High Value"
    }

    customer_df["Segment"] = customer_df["Cluster"].map(labels)

    return customer_df


# 🔹 7. Debug/Test
if __name__ == "__main__":
    from utils.preprocessing import load_data, add_profit

    df = load_data()
    df = add_profit(df)

    segmented = label_clusters(df)

    print(segmented.head())
    print("\nCluster Summary:\n", cluster_summary(df))