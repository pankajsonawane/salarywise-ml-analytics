
# SalaryWise ML Analytics

An interactive salary prediction and regression analytics
dashboard built using Python, Streamlit, Pandas, Plotly,
and Scikit-learn.

## Overview

SalaryWise predicts salary from years of professional experience
using a trained Linear Regression model.

The application also provides model evaluation, statistical
analysis, regression visualizations, and dataset exploration
through an interactive web dashboard.

## Features

- Salary prediction from professional experience
- Multiple experience scenario comparison
- Linear Regression slope, intercept, and equation
- Training and test R-squared scores
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Sum of Squares: SSR, SSE, and SST
- Actual vs. predicted salary visualization
- Residual analysis
- Mean, median, mode, variance, and standard deviation
- Skewness, histograms, and box plots
- Correlation matrix
- Dataset exploration and CSV downloads

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib

## Project Structure

    salarywise-ml-analytics/
    ├── app.py
    ├── regression_machine_learning.py
    ├── Salary_Data.csv
    ├── requirements.txt
    ├── README.md
    └── .gitignore

The trained model file is generated locally by the training script.

## Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Git

## Installation

Clone the repository:

    git clone https://github.com/pankajsonawane/salarywise-ml-analytics.git

Navigate into the project:

    cd salarywise-ml-analytics

Install dependencies:

    python -m pip install -r requirements.txt

## Train the Model

Run the training script to generate the model file:

    python regression_machine_learning.py

## Run the Application

Start the Streamlit application:

    streamlit run app.py

Open the local URL shown in your terminal.

## Model Evaluation

The dashboard evaluates the model on a held-out test set
and reports R-squared, MAE, MSE, and RMSE.

It also provides regression plots, residual diagnostics,
descriptive statistics, and sum-of-squares calculations.

## Important Notes

- Predictions are estimates, not guaranteed salaries.
- Results depend on the quality and representativeness
  of the training dataset.
- Predictions beyond the observed experience range may
  be unreliable.
- Only use trusted model files and datasets you have
  permission to share.

## Author

Your Name

## License

Add a license if you intend to grant others permission
to reuse, modify, and distribute your code.
