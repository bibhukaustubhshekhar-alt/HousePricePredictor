# HousePricePrediction

A machine learning project developed as part of my CSE core studies. This repository implements a regression pipeline to predict real estate prices based on physical house attributes.

## 📌 Overview
This project demonstrates a complete ML workflow:
1.  **Data Loading:** Reading raw CSV data using `pandas`.
2.  **Model Training:** Implementing a Linear Regression model with `scikit-learn`.
3.  **Validation:** Evaluating performance using the Root Mean Squared Error (RMSE) metric.
4.  **Model Persistence:** Saving/Loading models with `joblib`.
5.  **Inference:** A CLI tool for real-time price estimation.

---

## 📂 Project Structure
* `train.py` - Script to train the model and save the weights.
* `predict.py` - Script to load the model and take user input for predictions.
* `data/raw.csv` - The dataset containing house features and prices.
* `models/housing_model.pkl` - The trained model file (generated after training).

---

## 🚀 Installation & Usage

### 1. Requirements
Ensure you have the following Python libraries installed:
```bash
pip install pandas scikit-learn numpy joblib
```

### 2. Training
Place your dataset in the `data/` folder and run the training script:
```bash
python train.py
```
This will output the Validation RMSE and save the model to the `models/` directory.

### 3. Prediction
To use the trained model for predicting prices, run:
```bash
python predict.py
```
Enter the requested values (Bedrooms, Bathrooms, Sqft, etc.) when prompted.

---

## 🛠 Technical Details
* **Algorithm:** Linear Regression (Ordinary Least Squares).
* **Split Ratio:** 80% Training, 20% Validation.
* **Evaluation Metric:** RMSE (Root Mean Squared Error).
* **Input Features:** Bedrooms, Bathrooms, Sqft_living, Waterfront (0/1), and Floors.

---

## 📝 Future Scope
* Implement **Feature Scaling** (StandardScaler) for better convergence.
* Add **One-Hot Encoding** for categorical data like location.
* Deploy as a Web API using **FastAPI** or **Flask**.

--- 
**Project Type:** Artificial Intelligence & Machine Learning
```
