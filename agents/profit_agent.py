import pandas as pd


# 🔹 1. Calculate total profit per product
def profit_by_product(df):
    profit = df.groupby("Product_Name")["Profit"].sum().sort_values(ascending=False)
    return profit


# 🔹 2. Identify top profitable products
def top_profitable_products(df, top_n=5):
    profit = profit_by_product(df)
    return profit.head(top_n)


# 🔹 3. Identify loss-making products
def loss_products(df):
    profit = df.groupby("Product_Name")["Profit"].sum()
    loss = profit[profit < 0].sort_values()
    return loss


# 🔹 4. Total store profit
def total_profit(df):
    return df["Profit"].sum()


# 🔹 5. Profit insights (Agent intelligence)
def profit_insights(df):
    insights = []

    total = total_profit(df)
    top_products = top_profitable_products(df)
    loss = loss_products(df)

    # Overall profit status
    if total > 0:
        insights.append(f"Store is running in profit: {total}")
    else:
        insights.append(f"Store is running in loss: {total}")

    # Top products
    if not top_products.empty:
        insights.append("High-performing products are generating strong profit")

    # Loss products
    if not loss.empty:
        insights.append("Some products are causing losses → consider reducing stock or adjusting price")

    return insights


# 🔹 6. Profit vs Loss classification
def profit_status(df):
    df["Status"] = df["Profit"].apply(lambda x: "Profit" if x > 0 else "Loss")
    return df[["Product_Name", "Profit", "Status"]]


# 🔹 7. Main Agent Output
def profit_agent_summary(df):
    return {
        "total_profit": total_profit(df),
        "top_products": top_profitable_products(df),
        "loss_products": loss_products(df),
        "profit_status": profit_status(df).head(),
        "insights": profit_insights(df)
    }