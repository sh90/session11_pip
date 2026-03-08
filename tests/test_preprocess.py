import pandas as pd
from ml_project.preprocess import preprocess

def test_income_scaling(tmp_path):

    input_file = tmp_path/"input.csv"
    output_file = tmp_path/"output.csv"

    df = pd.DataFrame({
        "age":[30],
        "income":[100000],
        "purchased":[1]
    })
    df.to_csv(input_file,index=False)
    preprocess(input_file,output_file)

    result = pd.read_csv(output_file)

    assert result["income_scaled"][0] == 1
