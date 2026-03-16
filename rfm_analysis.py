import pandas as pd

df = pd.read_csv ("customer_transactions.csv")
print (df)

df["OrderDate"] = pd.to_datetime(df["OrderDate"])

print(df.head())

print(df.info())

print (df.isnull().sum())

print (df.describe())

customer_summary = df.groupby("CustomerID").agg({
    "OrderID" : "count",
    "Amount" : "sum",
    "OrderDate" : "max"
}).reset_index()

customer_summary.rename(columns={
    "OrderID": "TotalOrders",
    "Amount": "TotalSpend",
    "OrderDate": "LastPurchase"
}, inplace=True)

print (customer_summary)

today = df["OrderDate"].max()


customer_summary["Recency"] = (today - customer_summary["LastPurchase"]).dt.days
customer_summary["Frequency"] = customer_summary["TotalOrders"]
customer_summary["Monetary"] = customer_summary["TotalSpend"]
print(customer_summary)


customer_summary["R_Score"] = pd.qcut(
    customer_summary["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)
print(customer_summary)


customer_summary["F_Score"] = (
    customer_summary["Frequency"]
    .rank(method="dense", ascending=True)
    .astype(int)
)
print(customer_summary)

# Higher spend → higher M_Score
customer_summary["M_Score"] = (
    customer_summary["Monetary"]
    .rank(method="dense", ascending=False)
    .astype(int)
)
print(customer_summary)



customer_summary["RFM_Score"] = (
    customer_summary["R_Score"].astype(str) +
    customer_summary["F_Score"].astype(str) +
    customer_summary["M_Score"].astype(str)
)

print(customer_summary)



def segment_customer(row):
    r = int(row["R_Score"])
    f = int(row["F_Score"])
    m = int(row["M_Score"])

    if r >= 4 and f >= 4 and m >= 4:
        return "High Value"
    elif f >= 3 and r >= 3:
        return "Loyal"
    elif r >= 3:
        return "Potential"
    else:
        return "At Risk"
customer_summary["Segment"] = customer_summary.apply(
    segment_customer,
    axis=1
)

print(
    customer_summary[
        ["CustomerID", "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"]
    ]
)


print(customer_summary["Segment"].value_counts())

print(customer_summary[["CustomerID", "R_Score", "F_Score", "M_Score", "Segment"]])

print(customer_summary["Segment"].value_counts(normalize=True) * 100)



import matplotlib.pyplot as plt

customer_summary["Segment"].value_counts().plot(kind="bar")
plt.show()


customer_summary.to_csv("customer_rfm_final.csv", index=False)
