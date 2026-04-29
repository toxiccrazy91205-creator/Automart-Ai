import pandas as pd


# 🔹 1. Total product demand
def get_product_demand(df):
    demand = df.groupby("Product_Name")["Quantity_Sold"].sum().sort_values(ascending=False)
    return demand


# 🔹 2. Identify low-stock products (simple logic)
def low_stock_products(df, threshold=100):
    demand = df.groupby("Product_Name")["Quantity_Sold"].sum()

    low_stock = demand[demand < threshold].sort_values()
    return low_stock


# 🔹 3. Suggest restock quantity
def restock_suggestions(df):
    demand = df.groupby("Product_Name")["Quantity_Sold"].sum()

    suggestions = {}

    for product, qty in demand.items():
        if qty < 100:
            suggestions[product] = 150 - qty   # restock logic
        else:
            suggestions[product] = 0

    return suggestions


# 🔹 4. Festival-based demand (important feature ⭐)
def festival_demand(df):
    festival_df = df[df["Festival"] == 1]

    top_products = festival_df.groupby("Product_Name")["Quantity_Sold"].sum().sort_values(ascending=False)

    return top_products.head(5)


# 🔹 5. Inventory insights (Agent intelligence)
def inventory_insights(df):
    insights = []

    low_stock = low_stock_products(df)
    festival_top = festival_demand(df)

    if not low_stock.empty:
        insights.append("Some products are low in stock → restocking required")

    if not festival_top.empty:
        insights.append("Certain products have high demand during festivals → prepare stock in advance")

    return insights


# 🔹 6. Main Agent Output
def inventory_agent_summary(df):
    demand = get_product_demand(df)
    low_stock = low_stock_products(df)
    restock = restock_suggestions(df)
    festival_top = festival_demand(df)
    insights = inventory_insights(df)

    return {
        "top_demand_products": demand.head(),
        "low_stock_products": low_stock,
        "restock_suggestions": restock,
        "festival_top_products": festival_top,
        "insights": insights
    }