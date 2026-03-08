import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

THRESHOLD = 0.6
def evaluate():

    df = pd.read_csv("data/processed.csv")

    X = df[["age","income_scaled"]]
    y = df["purchased"]

    model = joblib.load("models/model.pkl")

    preds = model.predict(X)

    acc = accuracy_score(y,preds)

    if acc < THRESHOLD:
        raise ValueError("Model accuracy below threshold")

if __name__ == "__main__":
    evaluate()
