import pandas as pd

def load_iris_data(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded data from {path}")
        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None