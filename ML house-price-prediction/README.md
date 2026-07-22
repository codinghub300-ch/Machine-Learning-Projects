# House Price Prediction

A beginner-friendly Python machine learning project that predicts Egyptian
house prices with Linear Regression. It demonstrates a complete regression
workflow in one readable script.

## Project overview

The model estimates a property's price in Egyptian pounds (EGP) from `area_sq_m`,
bedrooms, bathrooms, age, distance from the city centre, city, and condition.

## Learning objectives

- Load and inspect a real-world-style CSV dataset.
- Detect and safely handle missing values.
- Prepare numeric and categorical features for a model.
- One-hot encode categories and prevent data leakage with a pipeline.
- Train and evaluate Linear Regression.
- Interpret MAE, MSE, RMSE, R-squared, and model coefficients.
- Create clear matplotlib charts and predict unseen homes.

## Dataset description

`dataset.csv` contains 220 synthetic-but-realistic property listings.
Prices vary by city, area, condition, age, distance to the centre, and normal
market variation. It is included for learning and demonstration, not as a
source of live market prices.

| Column | Description |
| --- | --- |
| `area_sq_m` | Property size in square metres |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Age of the property |
| `distance_to_center_km` | Distance from the city centre |
| `city` | Cairo, Giza, Alexandria, Mansoura, or New Cairo |
| `condition` | Fair, Good, or Excellent |
| `price_egp` | Target: sale price in Egyptian pounds |

A few intentional missing values demonstrate the pipeline's imputation step.

## Machine learning workflow

1. Load and validate the CSV data.
2. Explore shapes, types, missing values, statistics, and city counts.
3. Impute missing numeric values with the median and categories with the mode.
4. One-hot encode city and condition.
5. Split data into 80% training and 20% testing sets.
6. Train Linear Regression in a scikit-learn pipeline.
7. Evaluate test predictions with MAE, MSE, RMSE, and R-squared.
8. Print the strongest positive and negative coefficients.
9. Save four charts and predict prices for new properties.

## Folder structure

```text
house-price-prediction/
|-- main.py
|-- dataset.csv
|-- requirements.txt
|-- README.md
`-- charts/                    # Created or refreshed after running main.py
    |-- 01_price_distribution.png
    |-- 02_area_vs_price_by_city.png
    |-- 03_median_price_by_bedrooms.png
    `-- 04_actual_vs_predicted.png
```

## Installation

Use Python 3.10 or newer.

```bash
git clone <your-repository-url>
cd house-price-prediction
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The program prints progress, metrics, coefficients, and two sample
predictions. It saves four PNG charts in the `charts/` directory.

## Expected console output

```text
HOUSE PRICE PREDICTION WITH LINEAR REGRESSION

[1/7] Loading dataset
[2/7] Exploring the data
Rows and columns: (220, 8)
...
[5/7] Model evaluation
MAE:  EGP ...
MSE:  ...
RMSE: EGP ...
R-squared: ...

[6/7] Feature coefficients
...
[7/7] Creating charts
Saved 4/4 charts to: .../charts
Project completed successfully.
```

## Future improvements

- Add verified listings and more locations.
- Add parking, furnishing, floor, and nearby amenities.
- Compare Ridge, Random Forest, and Gradient Boosting models.
- Use cross-validation and hyperparameter tuning.
- Build an interactive Streamlit interface.

## License

Available for educational use.

---

---

<div align="center">

## 💙 Developed by Coding Hub

AI-Powered House Price Prediction System

© 2026 Coding Hub. All Rights Reserved.

</div>
