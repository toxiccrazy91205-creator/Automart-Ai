import pandas as pd
from models.kmeans_model import segment_customers
import os

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# 🔹 1. Get segmented customers
def get_customer_segments(df):
    customer_df = segment_customers(df)
    return customer_df


# 🔹 2. Identify high-value customers
def get_high_value_customers(df):
    customer_df = segment_customers(df)

    # cluster with highest average profit = high-value
    cluster_profit = customer_df.groupby("Cluster")["Profit"].mean()

    high_cluster = cluster_profit.idxmax()

    high_value_customers = customer_df[customer_df["Cluster"] == high_cluster]

    return high_value_customers


# 🔹 3. Identify low-value customers
def get_low_value_customers(df):
    customer_df = segment_customers(df)

    cluster_profit = customer_df.groupby("Cluster")["Profit"].mean()

    low_cluster = cluster_profit.idxmin()

    low_value_customers = customer_df[customer_df["Cluster"] == low_cluster]

    return low_value_customers


# 🔹 4. Generate marketing insights (Agent Decision)
def customer_insights(df):
    customer_df = segment_customers(df)

    cluster_summary = customer_df.groupby("Cluster").agg({
        "Quantity_Sold": "mean",
        "Profit": "mean"
    }).reset_index()

    # Use LangChain if available and API key is set
    if LANGCHAIN_AVAILABLE and "GEMINI_API_KEY" in os.environ:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            
            prompt = PromptTemplate.from_template(
                "You are an expert AI Marketing Agent for a supermarket. "
                "Analyze the following customer segments and provide 1-2 actionable, "
                "concise marketing strategies for each cluster based on their average profit and quantity sold.\n\n"
                "Customer Segments:\n{segments}\n\n"
                "Return the strategies as a clear bulleted list."
            )
            
            # Convert summary to a string format for the prompt
            segments_str = cluster_summary.to_string(index=False)
            
            chain = prompt | llm
            response = chain.invoke({"segments": segments_str})
            
            # Extract insights from the response
            insights = response.content.strip().split('\n')
            insights = [i.strip() for i in insights if i.strip()]
            return insights
        except Exception as e:
            print(f"⚠️ LangChain AI Insight generation failed: {e}. Falling back to rule-based insights.")

    # Fallback to simple rule-based insights
    insights = []

    for _, row in cluster_summary.iterrows():
        if row["Profit"] > 500:
            insights.append(f"Cluster {int(row['Cluster'])}: High-value customers → Offer loyalty rewards")
        elif row["Profit"] < 200:
            insights.append(f"Cluster {int(row['Cluster'])}: Low engagement → Provide discounts")
        else:
            insights.append(f"Cluster {int(row['Cluster'])}: Moderate customers → Target with promotions")

    return insights


# 🔹 5. Summary function (Main Agent Output)
def customer_agent_summary(df):
    high_value = get_high_value_customers(df)
    low_value = get_low_value_customers(df)
    insights = customer_insights(df)

    return {
        "high_value_customers": high_value.head(),
        "low_value_customers": low_value.head(),
        "insights": insights
    }