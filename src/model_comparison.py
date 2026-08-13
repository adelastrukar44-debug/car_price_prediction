import pandas as pd
import time

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)


DATA_PATH = "data/cleaned_cars_with_new_features.csv"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


print("Splitting features and target...")

x, y = split_features_and_target(df)


print("Creating train/test split...")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)


models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    ),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42,
    ),
}


results = []


for model_name, regressor in models.items():

    print(f"\nTraining {model_name}...")

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )



    start = time.time()

    model.fit(x_train, y_train)
    print(f"Training time: {time.time() - start:.2f} seconds")
    
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="mae",
    ascending=True,
)


print("\nModel comparison:")

print(results_df)
results_df.to_csv(
    "data/model_comparison_results.csv",
    index=False
)
