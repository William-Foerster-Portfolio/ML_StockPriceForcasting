# Google Stock Price Prediction Using Machine Learning Models

This project aims to predict Google stock prices using various machine learning models, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Unit (GRU) networks. The models are trained and evaluated using time series cross-validation with technical indicators and lag features.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Models and Features](#models-and-features)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Contributing](#contributing)

## Overview

This project leverages various deep learning models to forecast Google (GOOGL) stock prices based on historical data. The models incorporate several technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and more. The models are trained using time series cross-validation and evaluated using the Root Mean Squared Error (RMSE).

## Installation

To run this project, you need to have Python installed along with the necessary libraries. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/stock-price-prediction.git
    cd stock-price-prediction
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

## Usage

To run the project, execute the main script. The script will load the data, compute features, train the models, and visualize the results:

```bash
python main.py
