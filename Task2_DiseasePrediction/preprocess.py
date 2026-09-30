import pandas as pd

df = pd.read_csv("data/heart_disease.csv")

df = df.replace("?", pd.NA)

numeric_columns = [
    "age", "trestbps", "chol", "thalach",
    "oldpeak", "ca", "thal"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()

df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

df.to_csv("data/heart_disease_clean.csv", index=False)

print("Preprocessing completed!")
print("Shape:", df.shape)
print("\nTarget distribution:")
print(df["target"].value_counts())
print("\nMissing values:")
print(df.isnull().sum().sum())