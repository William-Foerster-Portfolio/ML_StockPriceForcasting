# Google Stock Price Prediction Using Machine Learning Models

This project aims to predict Google stock prices using various machine learning models, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Unit (GRU) networks. The models are trained and evaluated using time series cross-validation with technical indicators and lag features.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Models and Features](#models-and-features)
- [Results](#results)
- [Future Improvements](#fututure-improvements)

## Overview

This project leverages various deep learning models to forecast Google (GOOGL) stock prices based on historical data. The models incorporate several technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and more. The models are trained using time series cross-validation and evaluated using the Root Mean Squared Error (RMSE).

## Installation

To run this project, you need to have Python installed along with the necessary libraries. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone [https://github.com/will-foerster-portfolio/ML_StockPriceForcasting.git](https://github.com/William-Foerster-Portfolio/ML_StockPriceForcasting.git)
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

    **Note:** Ensure you have TensorFlow installed to run the neural network models:
    ```bash
    pip install tensorflow
    ```

## Data Preparation

The project uses historical stock price data for Google (GOOGL). Make sure you have a CSV file named `GOOGL_large.csv` in the root directory of the project. This CSV file should contain the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

The data is processed to compute technical indicators and time-based features before being used to train the models.

## Models and Features

### Technical Indicators and Time-Based Features

The following technical indicators and time-based features are used:
- **Technical Indicators**: Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Bollinger Bands, Money Flow Index (MFI), Average True Range (ATR), Force Index, Ease of Movement (EMV).
- **Time-Based Features**: Day of the week, quarter, month, year, day of the year, day of the month, week of the year.
- **Lagged Features**: Lagged values for high, low, open, and close prices for different time periods (day, week, month, quarter, year).

### Models

The following deep learning models are implemented:
- **CNN (Convolutional Neural Network)**: Captures local patterns in the time series data.
- **LSTM (Long Short-Term Memory Network)**: Captures long-term dependencies in time series data.
- **GRU (Gated Recurrent Unit)**: Similar to LSTM but with fewer parameters, which can speed up training while still capturing dependencies.

## Results

The performance of the models was evaluated using Root Mean Squared Error (RMSE) across multiple cross-validation folds. The results for each model are as follows:

- **CNN (Convolutional Neural Network) Model**: The CNN model was able to capture local patterns in the stock price data. The average RMSE across all folds for the CNN model is: `1.8298`.

- **LSTM (Long Short-Term Memory) Model**: The LSTM model, which is adept at capturing long-term dependencies in sequential data, demonstrated strong performance in predicting future stock prices. The average RMSE across all folds for the LSTM model is: `3.0450`.

- **GRU (Gated Recurrent Unit) Model**: The GRU model, similar to LSTM but more computationally efficient, also showed competitive performance in forecasting. The average RMSE across all folds for the GRU model is: `2.7762`.

Overall, all three models showed varying degrees of success in predicting stock prices, with CNN focusing on short-term patterns, while LSTM and GRU captured longer-term trends more effectively.

## Future Improvements

There are several potential areas for enhancing the models and the project overall:

### 1. Hyperparameter Tuning
   - **Objective**: Optimize model performance by fine-tuning hyperparameters such as the learning rate, number of layers, and number of units per layer.
   - **Approach**: Implement systematic hyperparameter optimization techniques like Grid Search, Random Search, or Bayesian Optimization to find the best configuration for each model.

### 2. Long-Term Forecasting
   - **Objective**: Extend the models' forecasting horizon to predict stock prices over longer periods (e.g., monthly or quarterly predictions).
   - **Approach**: Adjust sequence lengths and modify training data to support multi-step forecasting. Experiment with sequence-to-sequence models for predicting further into the future.

### 3. Feature Engineering
   - **Objective**: Enhance model inputs by creating additional features that could improve predictive performance.
   - **Approach**: Include more advanced technical indicators, external economic factors, news sentiment analysis, sector-specific financial metrics, company financials data to provide more context to the models.

### 4. Model Ensembling
   - **Objective**: Combine the strengths of multiple models to create a more robust ensemble model that can improve prediction accuracy.
   - **Approach**: Use ensemble techniques such as bagging, boosting, or stacking to aggregate predictions from CNN, LSTM, and GRU models.

### 5. Regularization and Dropout
   - **Objective**: Prevent overfitting and enhance model generalization to unseen data.
   - **Approach**: Introduce regularization techniques like L1 or L2 regularization and incorporate dropout layers to reduce the risk of overfitting during training.

### 6. Early Stopping and Learning Rate Schedules
   - **Objective**: Optimize the training process by preventing overtraining and ensuring efficient convergence.
   - **Approach**: Implement early stopping based on validation loss and employ learning rate schedules to dynamically adjust the learning rate during training.

### 7. Incorporating Attention Mechanisms
   - **Objective**: Improve the model's ability to focus on important features or time steps when making predictions.
   - **Approach**: Explore adding attention mechanisms to the LSTM and GRU models to better capture significant patterns and dependencies in the data.

By pursuing these improvements, we can enhance the models' accuracy and robustness, leading to more reliable stock price predictions.


