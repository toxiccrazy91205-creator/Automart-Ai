import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# 🔹 1. Load transaction data
def load_transaction_data():
    df = pd.read_csv("data/transaction_dataset.csv")
    return df


# 🔹 2. Convert to basket format
def create_basket(df):
    basket = df.pivot_table(
        index='Transaction_ID',
        columns='Product_Name',
        aggfunc=lambda x: 1,
        fill_value=0
    )
    return basket


# 🔹 3. Generate frequent itemsets
def generate_frequent_itemsets(basket, min_support=0.01):
    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    return freq_items


# 🔹 4. Generate association rules
def generate_rules(freq_items, min_lift=1):
    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    return rules


# 🔹 5. Main function (used by agents)
def get_rules():
    df = load_transaction_data()

    if df.empty:
        return pd.DataFrame()

    basket = create_basket(df)

    freq_items = generate_frequent_itemsets(basket)

    if freq_items.empty:
        return pd.DataFrame()

    rules = generate_rules(freq_items)

    return rules


# 🔹 6. Optional: Debug/Test function
if __name__ == "__main__":
    rules = get_rules()

    if rules.empty:
        print("No rules generated ❌")
    else:
        print("Association Rules:")
        print(rules.head())