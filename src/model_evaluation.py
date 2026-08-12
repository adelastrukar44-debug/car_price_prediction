import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_preprocessing import (
    split_features_and_target,
)


DATA_PATH = "data/cleaned_cars_with_new_features.csv"
MODEL_PATH = "models/car_price_model.joblib"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


print("Splitting features and target...")

x, y = split_features_and_target(df)


print("Creating the same train/test split...")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)


print("Loading trained model...")

model = joblib.load(MODEL_PATH)


print("Making predictions...")

y_pred = model.predict(x_test)

print("\nFirst 10 predictions:")
print(y_pred[:10])


print("\nCalculating regression metrics...")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})


print("\nRegression metrics:")
print(metrics)


prediction_analysis = pd.DataFrame({
    "price_usd": y_test.values,
    "predicted_price": y_pred,
})


prediction_analysis["error_price"] = (
    prediction_analysis["price_usd"]
    - prediction_analysis["predicted_price"]
)


prediction_analysis["absolute_error_price"] = (
    prediction_analysis["error_price"].abs()
)


print("\nPrediction examples:")

print(
    prediction_analysis.sample(
        10,
        random_state=42,
    )
)


print("\nLargest prediction errors:")

print(
    prediction_analysis
    .sort_values(
        "absolute_error_price",
        ascending=False,
    )
    .head(10)
)