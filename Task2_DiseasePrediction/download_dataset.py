import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df = pd.read_csv(url, header=None, names=columns)

df.to_csv("data/heart_disease.csv", index=False)

print("Dataset downloaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print(df.head())