import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

def load_data(path):
    data = pd.read_csv(path)
    return data

def build_features(data):
    features = data.drop(columns=["price"], errors="ignore")
    target = data["price"]
    return features, target

def train_model(features, target):
    train_X, valid_X, train_y, valid_y = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(train_X, train_y)
    predictions = model.predict(valid_X)
    rmse = np.sqrt(mean_squared_error(valid_y, predictions))
    return model, rmse

def save_model(model, path):
    joblib.dump(model, path)

if __name__ == "__main__":
    raw_path = "data/raw.csv"
    model_path = "models/housing_model.pkl"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    dataset = load_data(raw_path)
    features, target = build_features(dataset)
    model, validation_rmse = train_model(features, target)
    save_model(model, model_path)

    print(f"Training finished. Validation RMSE: {validation_rmse:.4f}")
    print(f"Model saved to: {model_path}")