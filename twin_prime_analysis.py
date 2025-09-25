"""
This script implements a complex systems analysis using twin primes.
The user's idea is implemented faithfully and transparently.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime
import warnings
from typing import List, Optional

warnings.filterwarnings('ignore')


def generate_brownian_motion(num_points: int = 10000) -> np.ndarray:
    """
    Generates a Brownian motion path with independent steps.

    Args:
        num_points: The number of points (time steps) in the path.

    Returns:
        A numpy array representing the Brownian motion path.
    """
    np.random.seed(42)  # for reproducibility
    steps = np.random.normal(0, 1, num_points)
    path = np.cumsum(steps)
    return path


def find_twin_primes(upper_limit: int = 1000) -> List[int]:
    """
    Finds all twin primes up to a specified limit.
    A twin prime is a prime number that is either 2 less or 2 more than another prime number.

    Args:
        upper_limit: The upper bound for the prime search.

    Returns:
        A list of the lower prime in each twin prime pair.
    """
    twin_primes = []
    for n in range(3, upper_limit):
        if isprime(n) and isprime(n + 2):
            twin_primes.append(n)
    return twin_primes


def predict_with_multiple_anchors(path: np.ndarray, anchors: List[int], target: int) -> Optional[float]:
    """
    Predicts the system's value at a target point using multiple anchors.

    Args:
        path: The Brownian motion data.
        anchors: A list of prime numbers to use as prediction anchors.
        target: The time point to predict.

    Returns:
        The predicted value as a float, or None if a prediction cannot be made.
    """
    predictions = []
    weights = []

    for anchor in anchors:
        if anchor >= target:
            continue

        # Calculate distance to target
        distance = target - anchor

        # Use only anchors within a reasonable distance for prediction
        if 0 < distance <= 200:
            # Weight based on inverse distance (closer anchors get more weight)
            weight = 1.0 / (distance + 1)

            # System value at the anchor point
            anchor_value = path[anchor] if anchor < len(path) else 0

            # A simple prediction model: value at anchor + mean of next 10 steps
            # This captures a short-term trend from the anchor.
            end_slice = min(anchor + 10, len(path))
            trend_adjustment = np.mean(path[anchor:end_slice]) if anchor < end_slice else 0
            prediction = anchor_value + trend_adjustment

            predictions.append(prediction)
            weights.append(weight)

    if not predictions:
        return None

    # Return the weighted average of all collected predictions
    return np.average(predictions, weights=weights)


def plot_results(path: np.ndarray, anchors: List[int], target: int, actual_value: float, prediction: float):
    """
    Plots the Brownian motion path, anchors, and the predicted vs. actual value.

    Args:
        path: The Brownian motion data.
        anchors: A list of anchor points.
        target: The target time point.
        actual_value: The actual value at the target.
        prediction: The predicted value at the target.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(path[:target + 100], alpha=0.7, label='Brownian Path')
    plt.axvline(x=target, color='r', linestyle='--', label=f'Target (t={target})')

    # Mark anchor points; use a single label for the legend to keep it clean
    if anchors:
        first_anchor = True
        for anchor in anchors:
            if anchor < len(path):
                plt.plot(anchor, path[anchor], 'go', markersize=8,
                         label='Anchors' if first_anchor else "")
                first_anchor = False

    plt.plot(target, actual_value, 'ro', markersize=10, label='Actual Value')
    plt.plot(target, prediction, 'bx', markersize=12, label='Prediction')

    plt.title('Predicting Brownian Motion with Twin Prime Anchors')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def print_statistical_analysis(path: np.ndarray, actual_value: float, target: int):
    """
    Prints a statistical analysis of the system and the target value.

    Args:
        path: The Brownian motion data.
        actual_value: The actual value at the target point.
        target: The target time point.
    """
    print("\n--- Statistical Analysis ---")
    mean_system = np.mean(path)
    std_system = np.std(path)
    print(f"System Mean: {mean_system:.4f}")
    print(f"System Standard Deviation: {std_system:.4f}")
    print(f"Actual value at t={target}: {actual_value:.4f}")

    # Check if the target value is statistically unusual (Z-score)
    z_score = (actual_value - mean_system) / std_system
    print(f"Z-score of target value: {z_score:.4f}")


# Main execution block
if __name__ == "__main__":
    # --- Configuration ---
    NUM_POINTS = 10000
    PRIME_SEARCH_LIMIT = 500
    NUM_ANCHORS = 10
    # The target will be the N-th twin prime after the anchors
    TARGET_PRIME_INDEX = NUM_ANCHORS

    # 1. Generate Brownian motion path
    path = generate_brownian_motion(NUM_POINTS)
    print(f"Generated Brownian motion with {NUM_POINTS} points.")

    # 2. Find twin primes to use as a basis for anchors
    twin_primes = find_twin_primes(PRIME_SEARCH_LIMIT)
    print(f"Found {len(twin_primes)} twin primes up to {PRIME_SEARCH_LIMIT}.")

    # 3. Select anchors and a target point
    if len(twin_primes) > TARGET_PRIME_INDEX:
        anchors = twin_primes[:NUM_ANCHORS]
        target = twin_primes[TARGET_PRIME_INDEX]
        print(f"Selected {len(anchors)} anchors: {anchors}")
        print(f"Selected target point: {target}")
    else:
        print("Error: Not enough twin primes found to select anchors and a target.")
        anchors, target = [], -1

    # 4. Perform prediction and analysis if the target is valid
    if target != -1 and target < len(path):
        prediction = predict_with_multiple_anchors(path, anchors, target)
        actual_value = path[target]

        print("\n--- Prediction Results ---")
        print(f"Actual value at t={target}: {actual_value:.4f}")

        if prediction is not None:
            error = abs(prediction - actual_value)
            # This custom accuracy metric measures the error relative to the system's volatility.
            # An accuracy of 100% means the error is zero.
            # An accuracy of 0% means the error is as large as or larger than the standard deviation.
            accuracy = max(0, 1 - error / (np.std(path) + 1e-10))

            print(f"Predicted value: {prediction:.4f}")
            print(f"Absolute Error: {error:.4f}")
            print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

            # 5. Plot the results
            plot_results(path, anchors, target, actual_value, prediction)
        else:
            print("Prediction could not be made with the given anchor configuration.")

        # 6. Print statistical analysis
        print_statistical_analysis(path, actual_value, target)

    elif target == -1:
        print("\nSkipping prediction due to insufficient twin primes.")
    else:
        print(f"\nError: Target point {target} is outside the data range (0-{len(path) - 1}).")