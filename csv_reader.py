import pandas as pd

def read_model():
    df = pd.read_csv('model.csv')
    return df

def read_data():
    df = pd.read_csv('iris.csv')
    # One hot encoding
    y = pd.get_dummies(df.species, prefix='Class')
    print(y.head())
    df["Class_1"] = y["Class_1"]
    df["Class_2"] = y["Class_2"]
    df["Class_3"] = y["Class_3"]
    print(df.head())
    return df
if __name__ == "__main__":
    print(read_model())
    print(read_data())