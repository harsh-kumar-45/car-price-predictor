"""Train the used-car price prediction pipeline.
Run: python train_model.py
"""
from pathlib import Path
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cleaned_Car_data.csv"
MODEL_PATH = BASE_DIR / "LinearRegressionModel.pkl"

FEATURES = ["name", "company", "year", "kms_driven", "fuel_type"]
TARGET = "Price"
CATEGORICAL = ["name", "company", "fuel_type"]
NUMERIC = ["year", "kms_driven"]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"Validation R2: {r2_score(y_test, predictions):.4f}")
    print(f"Validation MAE: {mean_absolute_error(y_test, predictions):.2f}")

    # Refit on all available data before saving the production model.
    pipeline.fit(X, y)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
