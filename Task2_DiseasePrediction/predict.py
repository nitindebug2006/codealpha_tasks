import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/heart_disease_clean.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Enter patient details:")

age = float(input("Age: "))
sex = float(input("Sex (1=Male, 0=Female): "))
cp = float(input("Chest pain type (0-3): "))
trestbps = float(input("Resting blood pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting blood sugar (0 or 1): "))
restecg = float(input("Resting ECG result (0-2): "))
thalach = float(input("Maximum heart rate: "))
exang = float(input("Exercise induced angina (0 or 1): "))
oldpeak = float(input("ST depression: "))
slope = float(input("Slope (0-2): "))
ca = float(input("Number of major vessels (0-3): "))
thal = float(input("Thalassemia (0-3): "))

patient = pd.DataFrame([[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]], columns=X.columns)

prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

print("\nPrediction:")

if prediction == 1:
    print("Disease detected")
else:
    print("No disease detected")

print("Disease Probability:", round(probability * 100, 2), "%")