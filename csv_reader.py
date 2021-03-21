import pandas as pd

def read_model():
    df = pd.read_csv('model.csv')
    return df

def read_data():
    df = pd.read_csv('iris.csv')
    return df
if __name__ == "__main__":
    print(read_model())
    print(read_data())