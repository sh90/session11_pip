import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

def train():
    df = pd.read_csv("data/processed.csv")
    X = df[["age","income_scaled"]]
    y = df["purchased"]
    model = LogisticRegression()
    model.fit(X,y)
    joblib.dump(model,"models/model.pkl")

if __name__ == "__main__":
    train()
