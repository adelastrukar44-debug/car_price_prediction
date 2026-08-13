
# Car Price Prediction

A machine learning project that predicts car prices based on various vehicle features using regression techniques.

## 📌 Project Overview

This project is a Machine Learning regression project for predicting used car prices based on vehicle characteristics.

The project covers the complete Machine Learning workflow:

* Data cleaning
* Missing value handling
* Feature engineering
* Data preprocessing
* Model training
* Model evaluation
* Model comparison

The goal is to build a model that can accurately predict the price of a used car.

---

## 📊 Dataset

The dataset contains information about used cars, including:

* Make
* Model
* Year
* Condition
* Mileage
* Fuel type
* Engine volume
* Color
* Transmission
* Drive unit
* Segment
* Price

The target variable is:

```text
price_usd
```

---

## 🧹 Data Cleaning

The data cleaning process includes:

* Standardizing column names
* Removing extra whitespace from text values
* Converting missing-like values to missing values
* Removing rows with missing target values
* Handling invalid engine volume and mileage values

---

## ⚙️ Feature Engineering

Additional features were created to improve model performance:

* `car_age`
* `engine_volume_liters`
* `mileage_per_year`
* `brand_model`

---

## 🔄 Data Preprocessing

Numerical features are processed using:

* Median imputation
* StandardScaler

Categorical features are processed using:

* Missing value imputation
* OneHotEncoder

The `condition` feature is treated as an ordinal feature using `OrdinalEncoder`.

All preprocessing steps are included in a Scikit-learn `ColumnTransformer` and combined with the regression model in a Pipeline.

---

## 🤖 Models

Four regression algorithms were tested:

* Random Forest Regressor
* Decision Tree Regressor
* Gradient Boosting Regressor
* Linear Regression

All models were evaluated using the same train/test split:

```text
Test size: 20%
Random state: 42
```

---

## 📈 Model Evaluation

The models were evaluated using:

* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

Lower MAE and RMSE values indicate better performance, while a higher R² indicates better performance.

---

## 🏆 Results

The Random Forest Regressor achieved the best overall performance.

| Model             |      MAE |     RMSE |        R² |
| ----------------- | -------: | -------: | --------: |
| **Random Forest** | **1060** | **2652** | **0.893** |
| Decision Tree     |     1380 |     3511 |     0.812 |
| Gradient Boosting |     1542 |     3104 |     0.853 |
| Linear Regression |     2078 |     4398 |     0.705 |

The Random Forest model explains approximately **89.3% of the variation in car prices** on the test dataset.

---

## 📁 Project Structure

```text
car_price_prediction/
│
├── data/
│   ├── cars.csv
│   ├── cleaned_cars_with_new_features.csv
│   └── model_comparison_results.csv
│
├── models/
│   └── car_price_model.joblib
│
├── notebooks/
│   └── model_comparison.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── model_evaluation.py
│   └── model_comparison.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> Note: Trained `.joblib` model files are excluded from Git because of their large file size.

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/adelastrukar44-debug/car_price_prediction.git
```

### 2. Navigate to the project directory

```bash
cd car_price_prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the data preprocessing and feature engineering

Run the corresponding scripts from the `src` directory to prepare the cleaned dataset.

### 7. Train the model

```bash
python src/train_model.py
```

This trains the Random Forest model and saves the trained model as:

```text
models/car_price_model.joblib
```

### 8. Evaluate the model

```bash
python src/model_evaluation.py
```

This calculates the regression metrics and displays prediction errors.

### 9. Compare models

```bash
python src/model_comparison.py
```

This trains and compares the four regression algorithms.

### 10. Explore the results

Open:

```text
notebooks/model_comparison.ipynb
```

to visualize and compare model performance.

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Jupyter Notebook

---

## 👤 Author

Adela Strukar
