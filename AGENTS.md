# Agent Instructions for the XAU/USD Pattern Prediction Model

This document provides instructions for an AI software engineer on how to build, train, and use the XAU/USD prediction model contained in this repository.

## 1. Model Overview

The prediction system consists of two main Python classes:

-   **`PatternAnalyzer` (`pattern_analyzer.py`)**: This is the core analysis engine. It is a general-purpose tool that takes sequences of patterns (e.g., `('U', 'D', 'U')` for Up-Down-Up movements) and analyzes their statistical properties. It uses a combination of sequence-based transition probabilities and historical frequency to make predictions. Its key features include combined predictions, partial pattern matching, and adaptive time weighting.

-   **`XauUsdPredictor` (`xauusd_predictor.py`)**: This is the application-specific class that uses the `PatternAnalyzer` to make predictions for the XAU/USD market. It handles data loading, processing OHLC data into the required pattern format, training the analyzer, and generating final trading signals (BUY, SELL, HOLD).

## 2. How to Build and Train the Model

Follow these steps to get the model ready for use.

### Step 2.1: Install Dependencies

The model requires the `pandas` library to process data. If it's not already installed, run the following command:

```bash
pip install pandas
```

### Step 2.2: Train the Model

The `xauusd_predictor.py` script is designed to be run directly. It will automatically load the training data, process it, and train the `PatternAnalyzer`.

To train the model, simply execute the script:

```bash
python3 xauusd_predictor.py
```

This will perform the training and then run a sample prediction based on the latest data in the training file.

## 3. How to Use Data from a Local MetaTrader Instance

The model is designed to work with historical price data exported from trading platforms like MetaTrader (MT4/MT5).

### Step 3.1: Export Data from MetaTrader

You need to instruct the user to export their historical data from their local MetaTrader terminal. The standard procedure is to use the "History Center" in MetaTrader to export the data for the desired symbol (e.g., XAUUSD) and timeframe (e.g., M1, H1).

The exported file **must be in CSV format** and placed in the root directory of this project.

### Step 3.2: Ensure Correct Data Format

The CSV file **must** contain the following columns in this order for the script to work correctly: `time`, `open`, `high`, `low`, `close`.

A sample header looks like this:
`time,open,high,low,close,tick_volume,spread,real_volume`

The model only requires the first five columns. The script is currently hardcoded to use the file named `XAUUSD_100k_M1_data.csv`. If the user provides a file with a different name, you must update the filename in the `if __name__ == '__main__':` block of `xauusd_predictor.py`.

## 4. How to Generate a Prediction and Signal

Once the model is trained, you can use the `XauUsdPredictor` class to generate signals for new, incoming data.

### Step 4.1: Load the Trained Model and New Data

First, create an instance of the predictor and train it on the historical dataset. Then, load the most recent data for which you want a prediction. The recent data should be a pandas DataFrame containing the last few data points (e.g., the last 3 candles for a pattern length of 3).

### Step 4.2: Generate the Signal

Use the `predict()` method to get a prediction dictionary, and then pass that result to the `generate_signal()` method.

Here is a complete code example:

```python
import pandas as pd
from xauusd_predictor import XauUsdPredictor

# 1. Initialize and train the predictor
predictor = XauUsdPredictor(config={
    "sequence_lengths": [3],
    "sequence_confidence_weight": 0.65,
    "frequency_confidence_weight": 0.35
})
predictor.train('XAUUSD_100k_M1_data.csv') # Use your historical data file

# 2. Get the most recent market data
#    In a live scenario, this would be the latest N candles from the market.
#    Here, we simulate it by taking data from the end of our dataset.
full_df = pd.read_csv('XAUUSD_100k_M1_data.csv')
# The number of rows should match the pattern length (e.g., 3)
recent_data = full_df.iloc[-3:].copy()

# 3. Make a prediction
prediction_result = predictor.predict(recent_data)
print("--- Prediction Result ---")
print(prediction_result)

# 4. Generate a trading signal
signal = predictor.generate_signal(prediction_result, confidence_threshold=0.4, delta_threshold=0.05)
print(f"\\n--- Generated Signal ---")
print(f"Signal: {signal}")
```

By following these steps, you can successfully use the model to generate trading signals based on data from a local MetaTrader instance.