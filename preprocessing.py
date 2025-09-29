from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    X = df.drop('species', axis=1)
    y = LabelEncoder().fit_transform(df['species'])
    return train_test_split(X, y, test_size=0.2, random_state=42)
