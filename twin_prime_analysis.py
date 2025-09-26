"""
This script now tests the "Twin Prime Anchoring" model on real-world financial data.
"""

import numpy as np
import pandas as pd
from sympy import isprime
import warnings
from typing import List, Optional, Dict, Any

warnings.filterwarnings('ignore')


# --- Data Loading Function ---

def load_financial_data(filepath: str, column: str = 'close') -> Optional[np.ndarray]:
    """
    Loads a specific column from a CSV file and returns it as a numpy array.

    Args:
        filepath: The path to the CSV file.
        column: The name of the column to extract.

    Returns:
        A numpy array of the data, or None if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)
        if column not in df.columns:
            print(f"Error: Column '{column}' not found in the CSV file.")
            return None
        return df[column].to_numpy()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


# --- Prime Number and Prediction Logic ---

def find_twin_primes(upper_limit: int) -> List[int]:
    """
    Finds all twin primes up to a specified limit.
    """
    twin_primes = []
    for n in range(3, upper_limit):
        if isprime(n) and isprime(n + 2):
            twin_primes.append(n)
    return twin_primes


def predict_with_multiple_anchors(path: np.ndarray, anchors: List[int], target: int) -> Optional[float]:
    """
    Predicts the system's value at a target point using a weighted average of anchor-based predictions.
    """
    predictions = []
    weights = []

    for anchor in anchors:
        if anchor >= target or anchor >= len(path):
            continue

        distance = target - anchor
        # We use a limited window for anchors to prevent using very old data
        if 0 < distance <= 500: # Increased window for potentially longer-term data
            weight = 1.0 / (distance + 1)
            anchor_value = path[anchor]

            # Simple trend adjustment based on a few subsequent data points
            end_slice = min(anchor + 10, len(path))
            trend_adjustment = np.mean(path[anchor:end_slice]) - anchor_value if end_slice > anchor else 0
            prediction = anchor_value + trend_adjustment

            predictions.append(prediction)
            weights.append(weight)

    if not predictions:
        return None

    return np.average(predictions, weights=weights)


# --- Main Analysis Function ---

def run_prediction_on_real_data(
    data_path: np.ndarray,
    prime_limit: int,
    num_anchors: int,
) -> Optional[Dict[str, Any]]:
    """
    Runs a single prediction task on the provided real-world data.
    """
    target_prime_index = num_anchors
    twin_primes = find_twin_primes(prime_limit)

    if len(twin_primes) <= target_prime_index:
        print("Error: Not enough twin primes found within the specified limit.")
        return None

    anchors = twin_primes[:num_anchors]
    target = twin_primes[target_prime_index]

    if target >= len(data_path):
        print(f"Error: Target index {target} is out of bounds for the data of length {len(data_path)}.")
        return None

    prediction = predict_with_multiple_anchors(data_path, anchors, target)
    if prediction is None:
        print("Model failed to make a prediction.")
        return None

    actual_value = data_path[target]
    error = abs(prediction - actual_value)
    # Accuracy is defined as 1 minus the error normalized by the data's standard deviation
    accuracy = max(0, 1 - error / (np.std(data_path) + 1e-10))

    return {
        "accuracy": accuracy,
        "error": error,
        "prediction": prediction,
        "actual": actual_value,
        "anchors": anchors,
        "target": target
    }


# --- Main Execution ---

def main():
    """
    Main function to load financial data and test the Twin Prime Anchoring model on it.
    """
    print("--- Testing Twin Prime Anchoring Model on Real Financial Data ---")

    # Configuration
    DATA_FILE = 'XAUUSD_100k_M1_data.csv'
    PRIME_SEARCH_LIMIT = 2000  # Increased limit to find more primes if needed
    NUM_ANCHORS = 25           # Using 25 anchors to predict the 26th

    # Load data
    price_path = load_financial_data(DATA_FILE, column='close')

    if price_path is not None:
        print(f"Successfully loaded {len(price_path)} data points from '{DATA_FILE}'.")

        # Run prediction
        result = run_prediction_on_real_data(
            data_path=price_path,
            prime_limit=PRIME_SEARCH_LIMIT,
            num_anchors=NUM_ANCHORS
        )

        # Report results
        if result:
            print("\n--- Prediction Results ---")
            print(f"Target Time Index (26th Twin Prime): {result['target']}")
            print(f"Predicted Value: {result['prediction']:.4f}")
            print(f"Actual Value:    {result['actual']:.4f}")
            print(f"Absolute Error:  {result['error']:.4f}")
            print(f"Model Accuracy:  {result['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()