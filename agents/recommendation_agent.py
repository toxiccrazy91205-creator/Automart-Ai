import pandas as pd
from models.apriori_model import get_rules


# 🔹 1. Get raw association rules
def get_all_rules():
    rules = get_rules()
    return rules


# 🔹 2. Clean and format rules
def formatted_rules():
    rules = get_rules()

    # convert frozenset → string
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]


# 🔹 3. Top recommendations
def top_recommendations(top_n=5):
    rules = get_rules()

    # sort by lift (importance)
    rules = rules.sort_values(by="lift", ascending=False)

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

    return rules[["antecedents", "consequents", "lift"]].head(top_n)


# 🔹 4. Recommend products for a given item
def recommend_for_product(product_name):
    rules = get_rules()

    recommendations = []

    for _, row in rules.iterrows():
        if product_name in row["antecedents"]:
            recommendations.append(list(row["consequents"]))

    return recommendations


# 🔹 5. Business insights (Agent intelligence)
def recommendation_insights():
    rules = get_rules()

    insights = []

    if not rules.empty:
        insights.append("Product bundling opportunities detected")
        insights.append("Cross-selling can increase revenue")

    return insights


# 🔹 6. Main Agent Output
def recommendation_agent_summary():
    return {
        "top_rules": top_recommendations(),
        "all_rules": formatted_rules().head(),
        "insights": recommendation_insights()
    }