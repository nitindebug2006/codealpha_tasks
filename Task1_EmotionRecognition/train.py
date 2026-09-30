import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/features.csv")

X = df.drop("emotion", axis=1)
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

if os.path.exists("models/emotion_model.pkl"):
    os.remove("models/emotion_model.pkl")
    print("Old model deleted")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

joblib.dump(
    model,
    "models/emotion_model.pkl"
)

loaded_model = joblib.load(
    "models/emotion_model.pkl"
)

classifier = loaded_model.named_steps["classifier"]

print()
print("Model Training Complete")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))
print("SVC Probability:", classifier.probability)
print(
    "Predict Probability Available:",
    hasattr(loaded_model, "predict_proba")
)
print("Model saved to: models/emotion_model.pkl")