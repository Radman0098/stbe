# Guide: Rebuilding the Twin Prime Anchor Prediction Model for an AI Assistant

## 1. Introduction
This guide provides step-by-step instructions to rebuild a time-series forecasting model based on a unique hypothesis: **"Twin primes can act as reliable anchor points in time-series data to predict future values."**

The objective is to create a Python script that reads time-series data (e.g., financial prices) from a CSV file and applies this model to forecast a future data point.

## 2. Prerequisites
Ensure the following Python libraries are installed. You can use `pip` to install them:
- `numpy`: For numerical operations.
- `pandas`: For reading and handling CSV data.
- `sympy`: For the `isprime` function to find prime numbers.

```bash
pip install numpy pandas sympy
```

## 3. Implementation Steps

### Step 1: Load the Time-Series Data
Create a Python function that takes a CSV file path and a column name as input. This function should:
1.  Read the CSV file using `pandas`.
2.  Extract the specified data column (e.g., `close`).
3.  Convert that column into a `numpy` array and return it.
4.  Implement error handling for cases where the file or column is not found.

**Example Implementation:**
```python
import pandas as pd
import numpy as np

def load_time_series_data(filepath: str, column: str = 'close') -> np.ndarray:
    """Loads a column from a CSV file into a numpy array."""
    try:
        df = pd.read_csv(filepath)
        if column not in df.columns:
            raise ValueError(f"Error: Column '{column}' not found in the CSV file.")
        return df[column].to_numpy()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")
```

### Step 2: Find Twin Primes
Write a function to find all twin primes up to a specified integer limit.
1.  A prime number `p` is a twin prime if `p + 2` is also prime.
2.  The function should take an integer `upper_limit` as input.
3.  Iterate from 3 up to the `upper_limit`, using `isprime` from the `sympy` library to check for twin prime pairs.
4.  Return a list containing the lower prime of each pair.

**Example Implementation:**
```python
from sympy import isprime
from typing import List

def find_twin_primes(upper_limit: int) -> List[int]:
    """Finds all twin primes up to a specified limit."""
    twin_primes = []
    for n in range(3, upper_limit):
        if isprime(n) and isprime(n + 2):
            twin_primes.append(n)
    return twin_primes
```

### Step 3: Implement the Core Prediction Logic
This function is the heart of the model. Its inputs are the data array, the list of twin primes (anchors), and the target index to be predicted.
1.  Generate a separate forecast from each anchor point.
2.  Calculate a weight for each forecast based on its "inverse distance" from the target (closer anchors get more weight).
3.  The prediction model for each anchor can be simple: **"the data value at the anchor + a small trend adjustment."** The trend adjustment can be calculated from the mean of a few data points immediately following the anchor.
4.  Finally, return the weighted average of all individual forecasts as the final prediction.

**Example Implementation:**
```python
def predict_with_anchors(path: np.ndarray, anchors: List[int], target: int) -> float:
    """Calculates a weighted prediction based on multiple anchor points."""
    predictions = []
    weights = []

    for anchor in anchors:
        if anchor >= target or anchor >= len(path):
            continue

        distance = target - anchor
        # Only consider anchors within a reasonable lookback window
        if 0 < distance <= 500:
            weight = 1.0 / (distance + 1)
            anchor_value = path[anchor]

            # Simple trend adjustment
            end_slice = min(anchor + 10, len(path))
            trend_adjustment = np.mean(path[anchor:end_slice]) - anchor_value
            prediction = anchor_value + trend_adjustment

            predictions.append(prediction)
            weights.append(weight)

    if not predictions:
        raise ValueError("Could not make a prediction with the given anchors.")

    return np.average(predictions, weights=weights)
```

### Step 4: Orchestrate and Report Results
Write a `main` function to coordinate the entire process:
1.  **Configure**: Define parameters such as the data file path, the prime search limit, and the number of anchors to use.
2.  **Load Data**: Use the function from Step 1 to load the dataset.
3.  **Find Anchors and Target**: Use the function from Step 2 to generate the list of twin primes. Use the first `N` primes as anchors and the `N+1`-th prime as the target index.
4.  **Run Prediction**: Call the prediction function from Step 3.
5.  **Calculate Accuracy**: A useful accuracy metric is `1 - (Absolute Error / Standard Deviation of Data)`. This normalizes the error and shows how much better the model is than a naive guess.
6.  **Report Results**: Print the predicted value, the actual value, and the final accuracy score.

## 4. Final Reference Code
Your complete, final script should look similar to the code below. This script combines all the steps into a single executable file.

```python
"""
This script tests the "Twin Prime Anchoring" model on real-world financial data.
"""

import numpy as np
import pandas as pd
from sympy import isprime
import warnings
from typing import List, Optional, Dict, Any

warnings.filterwarnings('ignore')

# --- Data Loading Function ---
def load_time_series_data(filepath: str, column: str = 'close') -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(filepath)
        if column not in df.columns:
            print(f"Error: Column '{column}' not found in the CSV file.")
            return None
        return df[column].to_numpy()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

# --- Prime Number and Prediction Logic ---
def find_twin_primes(upper_limit: int) -> List[int]:
    twin_primes = []
    for n in range(3, upper_limit):
        if isprime(n) and isprime(n + 2):
            twin_primes.append(n)
    return twin_primes

def predict_with_anchors(path: np.ndarray, anchors: List[int], target: int) -> Optional[float]:
    predictions = []
    weights = []
    for anchor in anchors:
        if anchor >= target or anchor >= len(path):
            continue
        distance = target - anchor
        if 0 < distance <= 500:
            weight = 1.0 / (distance + 1)
            anchor_value = path[anchor]
            end_slice = min(anchor + 10, len(path))
            trend_adjustment = np.mean(path[anchor:end_slice]) - anchor_value if end_slice > anchor else 0
            prediction = anchor_value + trend_adjustment
            predictions.append(prediction)
            weights.append(weight)
    if not predictions:
        return None
    return np.average(predictions, weights=weights)

# --- Main Analysis Function ---
def run_prediction_on_real_data(data_path: np.ndarray, prime_limit: int, num_anchors: int) -> Optional[Dict[str, Any]]:
    target_prime_index = num_anchors
    twin_primes = find_twin_primes(prime_limit)
    if len(twin_primes) <= target_prime_index:
        print("Error: Not enough twin primes found within the search limit.")
        return None

    anchors = twin_primes[:num_anchors]
    target = twin_primes[target_prime_index]

    if target >= len(data_path):
        print(f"Error: Target index {target} is out of bounds for data of length {len(data_path)}.")
        return None

    prediction = predict_with_anchors(data_path, anchors, target)
    if prediction is None:
        print("Model failed to make a prediction.")
        return None

    actual_value = data_path[target]
    error = abs(prediction - actual_value)
    accuracy = max(0, 1 - error / (np.std(data_path) + 1e-10))

    return {"accuracy": accuracy, "error": error, "prediction": prediction, "actual": actual_value, "target": target}

# --- Main Execution ---
def main():
    print("--- Testing Twin Prime Anchoring Model on Real Financial Data ---")

    # Configuration
    DATA_FILE = 'XAUUSD_100k_M1_data.csv'
    PRIME_SEARCH_LIMIT = 2000
    NUM_ANCHORS = 25

    # Load data and run prediction
    price_path = load_time_series_data(DATA_FILE, column='close')
    if price_path is not None:
        print(f"Successfully loaded {len(price_path)} data points from '{DATA_FILE}'.")
        result = run_prediction_on_real_data(data_path=price_path, prime_limit=PRIME_SEARCH_LIMIT, num_anchors=NUM_ANCHORS)

        # Report results
        if result:
            print("\n--- Prediction Results ---")
            print(f"Target Time Index ({NUM_ANCHORS + 1}-th Twin Prime): {result['target']}")
            print(f"Predicted Value: {result['prediction']:.4f}")
            print(f"Actual Value:    {result['actual']:.4f}")
            print(f"Absolute Error:  {result['error']:.4f}")
            print(f"Model Accuracy:  {result['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
```
By following this guide, an AI assistant will be able to accurately reconstruct the model and test its hypothesis on various datasets.