from src.data_loader import load_iris_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_confusion_matrix

path = r'D:\Coding\Iris-Flower-Classification\dataset\Iris.csv'

df = load_iris_data(path)
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_model(X_train, y_train)
y_pred = evaluate_model(model, X_test, y_test)
plot_confusion_matrix(y_test, y_pred, labels=df['species'].unique())
