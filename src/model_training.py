import joblib
import pandas as pd
import time
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
 
from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)

DATA_PATH = "data/cleaned_cars_with_new_features.csv"
MODEL_PATH = "models/car_price_model.joblib"

print("Loading dataset...")
 
df = pd.read_csv(DATA_PATH)

print("Splitting features and target...")
 
x, y = split_features_and_target(df)

print(x.shape)
print(y.shape)

print("Splitting data into training and test sets...")
 
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("Creating model pipeline...")
 
model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor( n_estimators=50,
                random_state=42,
                n_jobs=-1
            )),
    ]
)
print("Training model...")
start = time.time()
model.fit(x_train, y_train)
print(f"Training time: {time.time() - start:.2f} seconds")

print("Saving model...")

joblib.dump(model, MODEL_PATH)
 
print(f"Model saved to: {MODEL_PATH}")