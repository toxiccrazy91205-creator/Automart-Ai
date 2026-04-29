# 🚀 Main Entry Point for Supermarket AI System

from utils.preprocessing import preprocess_data

from models.lstm_pytorch import run_lstm
from models.kmeans_model import label_clusters, cluster_summary
from models.apriori_model import get_rules

from agents.inventory_agent import inventory_agent_summary
from agents.customer_agent import customer_agent_summary
from agents.profit_agent import profit_agent_summary
from agents.recommendation_agent import recommendation_agent_summary


def main():
    print("\n🚀 Starting Supermarket AI System...\n")

    # 🔹 Step 1: Load & preprocess data
    df = preprocess_data()
    print("✅ Data Loaded & Preprocessed Successfully\n")

    # 🔹 Step 2: Sales Prediction (LSTM)
    print("📈 Sales Prediction (LSTM)")
    try:
        prediction = run_lstm(df)
        print(f"➡️ Predicted Next Day Sales: {prediction:.2f}\n")
    except Exception as e:
        print("❌ LSTM Error:", e, "\n")

    # 🔹 Step 3: Customer Segmentation
    print("👥 Customer Segmentation (K-Means)")
    try:
        customers = label_clusters(df)
        print(customers.head(), "\n")

        print("📊 Cluster Summary:")
        print(cluster_summary(df), "\n")
    except Exception as e:
        print("❌ K-Means Error:", e, "\n")

    # 🔹 Step 4: Product Recommendations (Apriori)
    print("🛒 Product Recommendations (Apriori)")
    try:
        rules = get_rules()
        if rules.empty:
            print("⚠️ No rules generated\n")
        else:
            print(rules[["antecedents", "consequents", "lift"]].head(), "\n")
    except Exception as e:
        print("❌ Apriori Error:", e, "\n")

    # 🔹 Step 5: Inventory Agent
    print("📦 Inventory Agent")
    try:
        inventory = inventory_agent_summary(df)
        print("Top Demand Products:\n", inventory["top_demand_products"], "\n")
        print("Low Stock:\n", inventory["low_stock_products"], "\n")
        print("Insights:\n", inventory["insights"], "\n")
    except Exception as e:
        print("❌ Inventory Agent Error:", e, "\n")

    # 🔹 Step 6: Customer Agent
    print("👥 Customer Agent")
    try:
        customer = customer_agent_summary(df)
        print("High Value Customers:\n", customer["high_value_customers"], "\n")
        print("Insights:\n", customer["insights"], "\n")
    except Exception as e:
        print("❌ Customer Agent Error:", e, "\n")

    # 🔹 Step 7: Profit Agent
    print("💰 Profit Agent")
    try:
        profit = profit_agent_summary(df)
        print(f"Total Profit: {profit['total_profit']}\n")
        print("Top Products:\n", profit["top_products"], "\n")
        print("Loss Products:\n", profit["loss_products"], "\n")
        print("Insights:\n", profit["insights"], "\n")
    except Exception as e:
        print("❌ Profit Agent Error:", e, "\n")

    # 🔹 Step 8: Recommendation Agent
    print("🛒 Recommendation Agent")
    try:
        rec = recommendation_agent_summary()
        print("Top Rules:\n", rec["top_rules"], "\n")
        print("Insights:\n", rec["insights"], "\n")
    except Exception as e:
        print("❌ Recommendation Agent Error:", e, "\n")

    print("🎉 System Execution Completed Successfully!\n")


# 🔹 Run program
if __name__ == "__main__":
    main()