from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Create folder if it doesn't exist
    os.makedirs('assets', exist_ok=True)

    # 💾 Save model
    dump(model, 'assets/iris_model.joblib')
    print("✅ Model trained and saved to assets/iris_model.joblib")
    return model