"""Car Price Prediction project.

This script demonstrates a complete, beginner-friendly regression workflow:
loading data, exploring it, preparing features, training a linear regression
model, evaluating the model, visualizing results, and making sample
predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd


warnings.filterwarnings(
    "ignore",
    message="A NumPy version",
    category=UserWarning,
    module="sklearn.utils._param_validation",
)

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

TARGET_COLUMN = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class ModelArtifacts:
    """Container for objects used after training."""

    model: Pipeline
    feature_names: list[str]
    coefficients: list[float]
    intercept: float


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the dataset from disk with basic validation."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Make sure dataset.csv exists."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print a short data exploration summary."""

    print("\n=== Data Overview ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nBasic statistics:")
    print(df.describe(include="all").transpose())


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure the expected columns are present before modeling."""

    required_columns = {
        "year",
        "mileage",
        "engine_size",
        "horsepower",
        "doors",
        "owner_count",
        "brand",
        "model",
        "transmission",
        "fuel_type",
        "color",
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def create_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target."""

    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical columns."""

    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = features.select_dtypes(include=["object", "string"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )
    return preprocessor


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train a linear regression model inside a preprocessing pipeline."""

    preprocessor = build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate the model using standard regression metrics."""

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mse,
        "rmse": rmse,
        "r2": r2_score(y_test, predictions),
    }
    return metrics


def get_feature_names(model: Pipeline) -> list[str]:
    """Extract expanded feature names from the preprocessing pipeline."""

    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_coefficients(model: Pipeline) -> tuple[list[float], float]:
    """Read the trained linear regression coefficients and intercept."""

    regressor: LinearRegression = model.named_steps["regressor"]
    return regressor.coef_.tolist(), float(regressor.intercept_)


def build_artifacts(model: Pipeline) -> ModelArtifacts:
    """Create a bundle of model metadata for reporting."""

    feature_names = get_feature_names(model)
    coefficients, intercept = get_coefficients(model)
    return ModelArtifacts(
        model=model,
        feature_names=feature_names,
        coefficients=coefficients,
        intercept=intercept,
    )


def print_model_report(
    artifacts: ModelArtifacts,
    metrics: dict[str, float],
) -> None:
    """Display model evaluation results and coefficients."""

    print("\n=== Model Evaluation ===")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"MSE:  {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R^2:  {metrics['r2']:.4f}")

    coefficient_table = pd.DataFrame(
        {
            "feature": artifacts.feature_names,
            "coefficient": artifacts.coefficients,
        }
    ).sort_values(by="coefficient", key=lambda s: s.abs(), ascending=False)

    print("\n=== Top Coefficients ===")
    print(coefficient_table.head(10).to_string(index=False))
    print(f"\nIntercept: {artifacts.intercept:.2f}")


def save_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a simple bar chart with clear labels."""

    plt.figure(figsize=(10, 6))
    plt.bar(data[x_column], data[y_column], color="#2E86AB")
    plt.title(title)
    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_scatter_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a scatter chart for two numeric variables."""

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.65, color="#F18F01")
    plt.title(title)
    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a histogram for a numeric column."""

    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=20, color="#5B8E7D", edgecolor="black")
    plt.title(title)
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_boxplot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
) -> None:
    """Save a boxplot for categorical groups."""

    grouped = list(df.groupby(x_column))
    labels = [str(label) for label, _ in grouped]
    groups = [group[y_column].dropna().tolist() for _, group in grouped]

    plt.figure(figsize=(12, 6))
    plt.boxplot(groups, tick_labels=labels, patch_artist=True)
    plt.title(title)
    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_charts(df: pd.DataFrame) -> None:
    """Create and save all project charts."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    brand_summary = (
        df.groupby("brand", as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values(by=TARGET_COLUMN, ascending=False)
        .head(8)
    )
    save_bar_chart(
        brand_summary,
        x_column="brand",
        y_column=TARGET_COLUMN,
        title="Average Car Price by Brand",
        output_path=OUTPUT_DIR / "average_price_by_brand.png",
    )

    save_scatter_chart(
        df,
        x_column="mileage",
        y_column=TARGET_COLUMN,
        title="Car Price vs Mileage",
        output_path=OUTPUT_DIR / "price_vs_mileage.png",
    )

    save_histogram(
        df,
        column=TARGET_COLUMN,
        title="Distribution of Car Prices",
        output_path=OUTPUT_DIR / "price_distribution.png",
    )

    save_boxplot(
        df,
        x_column="transmission",
        y_column=TARGET_COLUMN,
        title="Car Price by Transmission Type",
        output_path=OUTPUT_DIR / "price_by_transmission.png",
    )


def prepare_sample_input() -> pd.DataFrame:
    """Create one realistic new sample for prediction."""

    sample = pd.DataFrame(
        [
            {
                "year": 2020,
                "mileage": 32000,
                "engine_size": 2.0,
                "horsepower": 180,
                "doors": 4,
                "owner_count": 1,
                "brand": "Toyota",
                "model": "Corolla",
                "transmission": "Automatic",
                "fuel_type": "Petrol",
                "color": "White",
            }
        ]
    )
    return sample


def predict_new_sample(model: Pipeline) -> None:
    """Generate a prediction for a new car listing."""

    sample = prepare_sample_input()
    predicted_price = float(model.predict(sample)[0])

    print("\n=== New Sample Prediction ===")
    print(sample.to_string(index=False))
    print(f"\nPredicted Price: ${predicted_price:,.2f}")


def main() -> None:
    """Run the full machine learning workflow."""

    print("Starting Car Price Prediction project...")

    try:
        df = load_dataset(DATASET_PATH)
        validate_required_columns(df)
        explore_data(df)

        print("\nCreating charts...")
        save_charts(df)
        print(f"Charts saved to: {OUTPUT_DIR}")

        print("\nPreparing training and testing data...")
        features, target = create_features_and_target(df)
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        print("Training Linear Regression model...")
        model = train_model(X_train, y_train)
        artifacts = build_artifacts(model)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print_model_report(artifacts, metrics)

        predict_new_sample(model)

        print("\nProject completed successfully.")

    except FileNotFoundError as exc:
        print(f"File error: {exc}")
    except ValueError as exc:
        print(f"Data error: {exc}")
    except NotFittedError as exc:
        print(f"Model error: {exc}")
    except Exception as exc:  # pragma: no cover - safety net for beginners
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
