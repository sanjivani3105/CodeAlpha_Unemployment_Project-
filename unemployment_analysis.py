import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Remove missing values
df = df.dropna()

# Display basic information
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nBasic Statistics:")
print(df.describe())

# -----------------------------------
# 1. Overall Unemployment Trend
# -----------------------------------

monthly = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Unemployment Rate Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------
# 2. Region-wise Unemployment
# -----------------------------------

region = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()
region = region.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
region.head(10).plot(kind="bar")
plt.title("Top 10 Regions by Average Unemployment Rate")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------
# 3. COVID-19 Impact
# -----------------------------------

before_covid = df[df["Date"] < "2020-03-01"]
during_covid = df[df["Date"] >= "2020-03-01"]

before_rate = before_covid["Estimated Unemployment Rate (%)"].mean()
during_rate = during_covid["Estimated Unemployment Rate (%)"].mean()

print("\nAverage Unemployment Before COVID-19:", round(before_rate, 2))
print("Average Unemployment During COVID-19:", round(during_rate, 2))

plt.figure(figsize=(7, 5))
plt.bar(
    ["Before COVID-19", "During COVID-19"],
    [before_rate, during_rate]
)
plt.title("Impact of COVID-19 on Unemployment")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# -----------------------------------
# 4. Urban vs Rural Analysis
# -----------------------------------

area = df.groupby("Area")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(7, 5))
area.plot(kind="bar")
plt.title("Urban vs Rural Unemployment")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------------
# 5. Correlation Heatmap
# -----------------------------------

numeric_data = df[
    [
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)"
    ]
]

plt.figure(figsize=(8, 5))
sns.heatmap(numeric_data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

