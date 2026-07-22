# Car Price Prediction

An educational Python machine learning project that predicts used car prices with **Linear Regression**. The goal is to show a complete, beginner-friendly regression workflow using clean code, clear console output, and practical charts.

## Project Overview

This project demonstrates how to:

- load a structured dataset
- inspect and understand the data
- handle missing values
- encode categorical features
- split data into training and test sets
- train a regression model
- evaluate model quality with standard metrics
- interpret coefficients
- generate predictions for a new car sample
- create useful visualizations

The project is intentionally simple so students can focus on the machine learning workflow instead of framework complexity.

## Learning Objectives

By studying this project, learners will understand:

- regression vs. classification
- how Linear Regression works at a practical level
- why preprocessing matters
- how one-hot encoding handles categorical data
- how to measure model performance
- how to read model coefficients
- how to present results in a professional way

## Dataset Description

The dataset is stored in `dataset.csv` and contains realistic sample car listings.

Columns include:

- `year`
- `mileage`
- `engine_size`
- `horsepower`
- `doors`
- `owner_count`
- `brand`
- `model`
- `transmission`
- `fuel_type`
- `color`
- `price`

The target variable is `price`.

The dataset contains around 220 rows and includes a few missing values in feature columns so the imputation step is meaningful.

## Machine Learning Workflow

`main.py` follows this pipeline:

1. Load the dataset.
2. Explore the data with summary statistics.
3. Handle missing values with imputers.
4. Encode categorical variables with one-hot encoding.
5. Split the data into training and testing sets.
6. Train a Linear Regression model.
7. Evaluate the model using MAE, MSE, RMSE, and R2.
8. Display feature coefficients.
9. Generate a prediction for a new sample car.
10. Save professional charts to the `outputs/` folder.

## Folder Structure

```text
car-price-prediction/
|-- dataset.csv
|-- main.py
|-- requirements.txt
|-- README.md
`-- outputs/
    |-- average_price_by_brand.png
    |-- price_by_transmission.png
    |-- price_distribution.png
    `-- price_vs_mileage.png
```

## Installation Instructions

1. Make sure Python 3.10+ is installed.
2. Open a terminal in the project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage Instructions

Run the project with:

```bash
python main.py
```

When the script runs, it will:

- print a data summary
- train the model
- print evaluation metrics
- show the most important coefficients
- predict the price of a sample car
- save charts in `outputs/`

## Expected Console Output

The exact numbers will vary slightly, but the output will look similar to this:

```text
Starting Car Price Prediction project...

=== Data Overview ===
Rows: 220
Columns: 12
...

Creating charts...
Charts saved to: outputs

Preparing training and testing data...
Training Linear Regression model...
Evaluating model...

=== Model Evaluation ===
MAE:  ...
MSE:  ...
RMSE: ...
R2:   ...

=== Top Coefficients ===
...

=== New Sample Prediction ===
...
Predicted Price: $...

Project completed successfully.
```

## Visualizations

The project saves at least four charts:

- `price_distribution.png`
- `price_vs_mileage.png`
- `average_price_by_brand.png`
- `price_by_transmission.png`

These charts help students connect the numbers to the story behind the data.

## Future Improvements

Possible next steps include:

- trying more advanced regression models
- adding cross-validation
- engineering new features such as car age
- tuning model hyperparameters
- comparing brands and fuel types more deeply
- building a small web app for interactive predictions

## Notes

- The dataset is synthetic but realistic enough for learning and portfolio use.
- The project is designed to be easy to read, easy to run, and easy to extend.
