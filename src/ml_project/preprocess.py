import pandas as pd

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    df["income_scaled"] = df["income"] / 100000
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess("data/raw.csv","data/processed.csv")
