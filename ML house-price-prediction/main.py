"""House Price Prediction: an educational Linear Regression project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset.csv"
CHARTS_DIR = PROJECT_DIR / "charts"
TARGET_COLUMN = "price_egp"

NUMERIC_FEATURES = [
    "area_sq_m",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_to_center_km",
]
CATEGORICAL_FEATURES = ["city", "condition"]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the project CSV file and check that it has usable rows."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Keep dataset.csv next to main.py."
        )

    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError("The dataset is empty. Add at least one data row.")

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
    return data


def explore_data(data: pd.DataFrame) -> None:
    """Print a compact, useful first look at the dataset."""
    print("\n[2/7] Exploring the data")
    print(f"Rows and columns: {data.shape}")
    print("\nColumn data types:")
    print(data.dtypes.to_string())
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print("\nNumeric summary:")
    print(data[NUMERIC_FEATURES + [TARGET_COLUMN]].describe().round(2).to_string())
    print("\nProperties by city:")
    print(data["city"].value_counts(dropna=False).to_string())


def build_model() -> Pipeline:
    """Create preprocessing and Linear Regression as one reproducible pipeline."""
    # Numeric gaps are filled with the median, which is robust to unusually
    # large or small homes. Categorical gaps use the most common category.
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # One-hot encoding converts labels such as Cairo into 0/1 columns.
            # Dropping one level per category avoids redundant dummy columns.
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer applies the appropriate preparation to each feature type.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # A Pipeline prevents data leakage: preprocessing is learned from training
    # data only when fit() is called.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def train_and_evaluate(
    data: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, np.ndarray]:
    """Split data, fit the model, and print regression evaluation metrics."""
    print("\n[3/7] Preparing features and splitting data")
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data[TARGET_COLUMN]

    # The test set is kept unseen during fitting to give an honest evaluation.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )
    print(f"Training rows: {len(x_train)} | Testing rows: {len(x_test)}")

    print("\n[4/7] Training Linear Regression model")
    model = build_model()
    model.fit(x_train, y_train)

    # Predictions on the test set are compared with the true held-out prices.
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r_squared = r2_score(y_test, predictions)

    print("\n[5/7] Model evaluation")
    print(f"MAE:  EGP {mae:,.2f}")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: EGP {rmse:,.2f}")
    print(f"R²:   {r_squared:.4f}")
    return model, x_test, y_test, predictions


def display_coefficients(model: Pipeline) -> None:
    """Show the strongest positive and negative Linear Regression coefficients."""
    print("\n[6/7] Feature coefficients")
    preprocessor = model.named_steps["preprocessor"]
    regression = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = pd.Series(regression.coef_, index=feature_names).sort_values()

    print("Largest negative coefficients:")
    print(coefficients.head(5).round(2).to_string())
    print("\nLargest positive coefficients:")
    print(coefficients.tail(5).round(2).to_string())
    print(
        "\nInterpretation: after other features are held constant, a positive "
        "coefficient raises the estimated price and a negative one lowers it."
    )


def save_chart(figure: plt.Figure, filename: str) -> bool:
    """Save one chart without preventing the rest of the workflow from running."""
    try:
        figure.savefig(CHARTS_DIR / filename, dpi=150)
        return True
    except OSError as error:
        print(f"Could not save {filename}: {error}")
        return False


def create_visualizations(
    data: pd.DataFrame,
    actual: pd.Series,
    predicted: np.ndarray,
) -> None:
    """Create and save four clear charts that explain the data and model."""
    print("\n[7/7] Creating charts")
    try:
        CHARTS_DIR.mkdir(exist_ok=True)
    except OSError as error:
        print(f"Could not create the charts folder: {error}")
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    saved_charts = 0

    # Chart 1: target distribution helps identify the typical price range.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data[TARGET_COLUMN], bins=18, color="#2E86AB", edgecolor="white")
    ax.set_title("Distribution of House Prices")
    ax.set_xlabel("Price (EGP)")
    ax.set_ylabel("Number of Properties")
    fig.tight_layout()
    saved_charts += save_chart(fig, "01_price_distribution.png")
    plt.close(fig)

    # Chart 2: compares area-price patterns across the major locations.
    fig, ax = plt.subplots(figsize=(9, 5))
    for city, group in data.dropna(subset=["city", "area_sq_m"]).groupby("city"):
        ax.scatter(
            group["area_sq_m"],
            group[TARGET_COLUMN],
            label=city,
            alpha=0.75,
            s=38,
        )
    ax.set_title("Area and Price by City")
    ax.set_xlabel("Area (square metres)")
    ax.set_ylabel("Price (EGP)")
    ax.legend(title="City")
    fig.tight_layout()
    saved_charts += save_chart(fig, "02_area_vs_price_by_city.png")
    plt.close(fig)

    # Chart 3: median prices make the bedroom relationship easy to compare.
    median_prices = data.groupby("bedrooms")[TARGET_COLUMN].median()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        median_prices.index.astype(str),
        median_prices.values,
        color="#F18F01",
        label="Median price",
    )
    ax.set_title("Median House Price by Bedroom Count")
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Median Price (EGP)")
    ax.legend()
    fig.tight_layout()
    saved_charts += save_chart(fig, "03_median_price_by_bedrooms.png")
    plt.close(fig)

    # Chart 4: points near the diagonal show accurate test-set predictions.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, color="#6A4C93", alpha=0.8, label="Predictions")
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(bounds, bounds, "r--", linewidth=2, label="Perfect prediction")
    ax.set_title("Actual vs Predicted Test Prices")
    ax.set_xlabel("Actual Price (EGP)")
    ax.set_ylabel("Predicted Price (EGP)")
    ax.legend()
    fig.tight_layout()
    saved_charts += save_chart(fig, "04_actual_vs_predicted.png")
    plt.close(fig)

    print(f"Saved {saved_charts}/4 charts to: {CHARTS_DIR}")


def predict_new_homes(model: Pipeline) -> None:
    """Estimate prices for two new examples using the fitted pipeline."""
    new_homes = pd.DataFrame(
        [
            {
                "area_sq_m": 145,
                "bedrooms": 3,
                "bathrooms": 2,
                "age_years": 6,
                "distance_to_center_km": 10,
                "city": "Cairo",
                "condition": "Good",
            },
            {
                "area_sq_m": 220,
                "bedrooms": 4,
                "bathrooms": 3,
                "age_years": 2,
                "distance_to_center_km": 7,
                "city": "New Cairo",
                "condition": "Excellent",
            },
        ]
    )
    new_homes["predicted_price_egp"] = model.predict(new_homes)

    print("\nSample predictions for new homes:")
    for index, row in new_homes.iterrows():
        print(
            f"  Home {index + 1}: {row['area_sq_m']:.0f} m², {row['city']} "
            f"-> EGP {row['predicted_price_egp']:,.0f}"
        )


def main() -> None:
    """Run the complete educational machine learning workflow."""
    print("=" * 62)
    print("HOUSE PRICE PREDICTION WITH LINEAR REGRESSION")
    print("=" * 62)

    try:
        print("\n[1/7] Loading dataset")
        data = load_dataset(DATASET_PATH)
        explore_data(data)
        model, x_test, y_test, predictions = train_and_evaluate(data)
        display_coefficients(model)
        create_visualizations(data, y_test, predictions)
        predict_new_homes(model)
        print("\nProject completed successfully.")
    except FileNotFoundError as error:
        print(f"\nFile error: {error}")
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"\nData error: {error}")
    except Exception as error:
        print(f"\nUnexpected error: {error}")
        print("Check requirements.txt is installed and dataset.csv is valid.")


if __name__ == "__main__":
    main()
