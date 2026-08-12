import pandas as pd
import numpy as np
from datetime import datetime

# Create new columns
def _create_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
     
    df["car_age"] = datetime.now().year - df["year"]
    df["engine_volume_liters"] = df["volume_cm3"] / 1000
    df["mileage_per_year"] = (df["mileage_km"] / df["car_age"].replace(0, np.nan))
    df["brand_model"] = df["make"] + "_" + df["model"]
    
    return df

# Build features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
 
    df_features = (
        df
        .pipe(_create_new_columns)
        
    )
 
    return df_features


CLEANED_DATA_PATH = "data/cleaned_cars.csv"
FEATURES_DATA_PATH = "data/cleaned_cars_with_new_features.csv"

def main() -> None:
    """Load cleaned data, build features, and save the feature-engineered dataset."""
    print("Loading cleaned dataset...")
 
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
 
    print("Building features...")
 
    df_features = build_features(df_cleaned)
 
    print("Saving feature-engineered dataset...")
 
    df_features.to_csv(FEATURES_DATA_PATH, index=False)
 
    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")


if __name__ == "__main__": 
    main()
    # python -m src.feature_engineering 

 