# Linear Regression with Gradient Descent

This repository contains an implementation of Linear Regression using Gradient Descent in Python. It includes the ability to fit a linear model, evaluate its performance using various metrics, visualize the cost function over iterations, and apply K-fold cross-validation for model evaluation.

## Features

- **Linear Regression Model**: Implemented using Gradient Descent to optimize weights.
- **Performance Metrics**: Calculates MSE, RMSE, MAE, R-squared, Adjusted R-squared, SSR, SSE, and SST.
- **Model Visualization**: Plot the MSE vs Iterations graph for cost function analysis.
- **Cross-validation**: K-fold cross-validation with different performance metrics to assess the model.
- **Hyperparameters**: Adjustable learning rate, number of iterations, and binning for MSE history storage.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages using `pip`:

```bash
pip install numpy matplotlib scikit-learn
