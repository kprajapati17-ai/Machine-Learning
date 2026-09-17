import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


transactions = [
    ["Bread", "Milk"],
    ["Bread", "Milk"],
    ["Bread", "Milk"],
    ["Bread", "Milk"],
    ["Bread", "Milk"],
    ["Bread", "Milk"],
    ["Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk"],
    ["Milk"],
    ["Butter"],
    ["Butter"],
    ["Butter"],
    ["Butter"],
    ["Butter"]
]


encoder = TransactionEncoder()
data =encoder.fit_transform(transactions)
df = pd.DataFrame(data,columns=encoder.columns_)

frequent_itemsets = apriori(
    df,
    min_support=0.40,
    use_colnames=True
)

print("\n===== FREQUENT ITEMSETS =====")
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.50
)

print("\n===== ASSOCIATION RULES =====")

for _, rule in rules.iterrows():

    print(
        f"{rule['antecedents']} -> "
        f"{rule['consequents']}"
    )

    print(f"Support: {rule['support']:.2%}")
    print(f"Confidence: {rule['confidence']:.2%}")
    print(f"Lift: {rule['lift']:.2f}")
    print("-" * 30)

strong_rules = rules[
    (rules["confidence"] >= 0.60) &
    (rules["lift"] > 1)
]

print("\n===== STRONG RULES =====")

for _, rule in strong_rules.iterrows():

    print(
        f"{rule['antecedents']} -> "
        f"{rule['consequents']}"
    )

    print(f"Support: {rule['support']:.2%}")
    print(f"Confidence: {rule['confidence']:.2%}")
    print(f"Lift: {rule['lift']:.2f}")
    print("-" * 30)