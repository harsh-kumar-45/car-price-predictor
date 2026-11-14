from pathlib import Path
import os
import pickle

import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)

with (BASE_DIR / "LinearRegressionModel.pkl").open("rb") as model_file:
    model = pickle.load(model_file)

car = pd.read_csv(BASE_DIR / "Cleaned_Car_data.csv")


@app.route("/", methods=["GET"])
def index():
    companies = sorted(car["company"].dropna().unique().tolist())
    car_models = sorted(car["name"].dropna().unique().tolist())
    years = sorted(car["year"].dropna().unique().tolist(), reverse=True)
    fuel_types = sorted(car["fuel_type"].dropna().unique().tolist())

    companies.insert(0, "Select Company")
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        company = request.form.get("company", "").strip()
        car_model = request.form.get("car_models", "").strip()
        year = request.form.get("year", "").strip()
        fuel_type = request.form.get("fuel_type", "").strip()
        driven = request.form.get("kilo_driven", "").strip()

        if not all([company, car_model, year, fuel_type, driven]):
            return "Please fill in all fields", 400
        if company == "Select Company":
            return "Please select a company", 400

        year = int(year)
        driven = float(driven)
        if driven < 0:
            return "Kilometres driven cannot be negative", 400

        input_data = pd.DataFrame(
            [[car_model, company, year, driven, fuel_type]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
        )
        prediction = float(model.predict(input_data)[0])
        return str(round(prediction, 2))

    except (TypeError, ValueError):
        return "Invalid input values", 400
    except Exception:
        app.logger.exception("Prediction failed")
        return "Prediction failed. Please try again later.", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
