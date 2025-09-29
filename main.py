from data_loader import load_iris_data
from preprocessing import preprocess_data
from model import train_model
from evaluate import evaluate_model
from visualize import plot_confusion_matrix

# ✅ Load the dataset
path = r'D:\Coding\Iris-Flower-Classification\dataset\Iris.csv'
df = load_iris_data(path)  # ← This line defines df

# ✅ Preprocess and train
(X_train, X_test, y_train, y_test), df_clean = preprocess_data(df)
model = train_model(X_train, y_train)
y_pred = evaluate_model(model, X_test, y_test)
plot_confusion_matrix(y_test, y_pred, labels=df_clean['species'].unique())
