import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Target column is price_usd
TARGET_COLUMN = "price_usd"

# Numerical columns
NUMERIC_FEATURES = [
    "year",
    "mileage_km",
    "volume_cm3",
    "car_age"
    "engine_volume_liters",
    "mileage_per_year"
]

# Categorical columns
CATEGORICAL_FEATURES = [
    "make",
    "model",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
    "brand_model"
]

# Ordinal column
ORDINAL_FEATURES = ["condition"]

# Combining all feature transformations
def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES


# Numerical transformer
def _build_numeric_transformer() -> Pipeline:
 
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
 
    return numeric_transformer

# Categorical transformer
def _build_categorical_transformer() -> Pipeline:
 
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
 
    return categorical_transformer

# Ordinal transformer
def _build_ordinal_transformer() -> Pipeline:
 
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[
                    ["for parts", "with damage", "with mileage"] 
                ]
            )),
        ]
    )
 
    return ordinal_transformer


# Column transformer
def build_preprocessor() -> ColumnTransformer:
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", _build_numeric_transformer(), NUMERIC_FEATURES),
            ("cat", _build_categorical_transformer(), CATEGORICAL_FEATURES),
            ("ord", _build_ordinal_transformer(), ORDINAL_FEATURES),
        ],
        remainder="drop"
    )
 
    return preprocessor


