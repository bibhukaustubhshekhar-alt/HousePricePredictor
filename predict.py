import pandas as pd
import joblib


def load_model(path):
    model = joblib.load(path)
    return model


def prompt_int(prompt_text):
    while True:
        raw = input(prompt_text).strip()
        try:
            return int(raw)
        except ValueError:
            print("Invalid integer, please try again.")


def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    model_path = "models/housing_model.pkl"
    model = load_model(model_path)

    print("Enter house features for price prediction")
    bedrooms = prompt_int("Bedrooms (e.g., 3): ")
    bathrooms = prompt_int("Bathrooms (e.g., 2): ")
    sqft_living = prompt_int("Square feet living area (e.g., 1500): ")
    waterfront = prompt_int("Waterfront (0 or 1): ")
    floors = prompt_int("Floors (1-3): ")

    sample = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "waterfront": waterfront,
        "floors": floors,
    }])

    prediction = make_predictions(model, sample)[0]
    print(f"Predicted house price: ${prediction:,.2f}")