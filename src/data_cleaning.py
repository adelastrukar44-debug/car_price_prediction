import re
import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cleaned_cars.csv"


# Standardize Columns
def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    new_columns = []
 
    for col in df.columns:
        clean_col = col.strip().lower()
 
        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")
 
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")
 
        new_columns.append(clean_col)
 
    df.columns = new_columns
    return df

# Removing Extra Whitespace from Text Values
def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    text_columns = df.select_dtypes(include=["str"]).columns
 
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
 
    return df


MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}

# Replace missing_like_values with NA
def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
 
    return df

# Fix invalid volume
def _fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[df["volume_cm3"] >= 10000, "volume_cm3"] = pd.NA
    df.loc[df["mileage_km"] == 9999999.0, "mileage_km"] = pd.NA

    return df

# Remove rows with missing values in target columns
def _remove_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.dropna(subset=["price_usd"])
 
    return df

# Pipeline
def clean(df: pd.DataFrame) -> pd.DataFrame:
 
    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_replace_missing_like_values)
        .pipe(_fix_invalid_values)
        .pipe(_remove_rows_with_missing_target)
        .reset_index(drop=True)
    )
 
    return df_clean

# Finish cleaning and save cleaned dataset
def main() -> None:
    """Load raw data, clean it, and save the cleaned dataset."""
    print("Loading raw dataset...")
 
    df_raw = pd.read_csv(RAW_DATA_PATH)
 
    print("Cleaning dataset...")
 
    df_cleaned = clean(df_raw)
 
    print("Saving cleaned dataset...")
 
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
 
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    
if __name__ == "__main__":
    main()