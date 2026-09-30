import pandas as pd

df = pd.read_csv("data/heart_disease.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nTarget Values:")
print(df["target"].unique())