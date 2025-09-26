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


def generate_brownian_motion(
    num_points: int = 10000,
    drift: float = 0.0,
    volatility: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Brownian motion path with customizable drift and volatility.

    Args:
        num_points: The number of points (time steps) in the path.
        drift: The constant drift (trend) for each step.
        volatility: The standard deviation of the random steps.
        seed: An optional random seed for reproducibility. If None, the path is random.

    Returns:
        A numpy array representing the Brownian motion path.
    """
    if seed is not None:
        np.random.seed(seed)

    # The random steps (innovations)
    random_steps = np.random.normal(0, 1, num_points)

    # The steps with drift and volatility applied
    # Each step is trend + random_fluctuation
    steps = drift + volatility * random_steps

    # The cumulative sum creates the path
    path = np.cumsum(steps)
    return path


def generate_chaotic_path(num_points: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generates a chaotic, non-stationary path where volatility and drift change over time.

    This is significantly more complex than a standard Brownian motion.
    - Volatility follows its own random walk with mean reversion.
    - Drift randomly flips its sign and magnitude at intervals.

    Args:
        num_points: The number of points for the path.
        seed: An optional random seed for reproducibility.

    Returns:
        A numpy array representing the chaotic path.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Dynamic Volatility (follows a smoothed, mean-reverting random walk)
    volatility_path = np.zeros(num_points)
    current_volatility = 1.5
    # Innovations for the volatility's own random walk
    volatility_innovations = np.random.normal(0, 0.2, num_points)
    for i in range(num_points):
        # Add a random shock, but pull it back towards a mean of 1.5 to prevent it from exploding or vanishing
        current_volatility = (0.995 * current_volatility) + (0.005 * 1.5) + volatility_innovations[i]
        volatility_path[i] = max(0.2, current_volatility)  # Ensure volatility stays positive

    # 2. Flipping Drift (changes sign and magnitude randomly)
    drift_path = np.zeros(num_points)
    current_drift = 0.05
    # On average, change drift every 2000 steps
    change_probability = 1 / 2000
    for i in range(num_points):
        if np.random.random() < change_probability:
            # Flip sign and select a new random magnitude
            current_drift = np.random.normal(0, 0.15)
        drift_path[i] = current_drift

    # 3. Generate the final path using the dynamic, chaotic parameters
    path_steps = np.zeros(num_points)
    random_innovations = np.random.normal(0, 1, num_points)
    for i in range(num_points):
        # The step at time 'i' is determined by the drift and volatility at time 'i'
        path_steps[i] = drift_path[i] + volatility_path[i] * random_innovations[i]

    return np.cumsum(path_steps)


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


def run_single_simulation(
    num_points: int,
    prime_limit: int,
    num_anchors: int,
    show_plot: bool = False,
    seed: Optional[int] = None
) -> Optional[dict]:
    """
    Runs a single, complete simulation using a chaotic path.
    """
    target_prime_index = num_anchors

    # 1. Generate a chaotic path
    path = generate_chaotic_path(num_points, seed)

    # 2. Find primes
    twin_primes = find_twin_primes(prime_limit)

    # 3. Select anchors and target
    if len(twin_primes) <= target_prime_index:
        return None  # Not enough primes

    anchors = twin_primes[:num_anchors]
    target = twin_primes[target_prime_index]

    if target >= num_points:
        return None  # Target is out of bounds

    # 4. Predict
    prediction = predict_with_multiple_anchors(path, anchors, target)
    if prediction is None:
        return None  # Prediction failed

    actual_value = path[target]
    error = abs(prediction - actual_value)
    accuracy = max(0, 1 - error / (np.std(path) + 1e-10))

    if show_plot:
        print(f"--- Simulation Run (Plotting) ---")
        print(f"Actual: {actual_value:.2f}, Predicted: {prediction:.2f}, Error: {error:.2f}, Accuracy: {accuracy*100:.1f}%")
        plot_results(path, anchors, target, actual_value, prediction)
        print_statistical_analysis(path, actual_value, target)

    return {"accuracy": accuracy, "error": error}


# Main execution block
if __name__ == "__main__":
    # --- Revised Ultimate Stress Test ---
    # After hitting the environment's resource limits, this test is scaled
    # to be the maximum challenge that can be reliably completed.

    NUM_SIMULATIONS = 200      # Reduced from 1000
    PATH_LENGTH = 50000        # Reduced from 200,000
    PRIME_SEARCH_LIMIT = 1000  # Reduced from 2000
    NUM_ANCHORS = 25           # Reduced from 50

    print(f"--- Starting Revised Ultimate Stress Test ---")
    print(f"Running {NUM_SIMULATIONS} simulations on chaotic paths of length {PATH_LENGTH}.")
    print(f"Using {NUM_ANCHORS} anchors, searching for primes up to {PRIME_SEARCH_LIMIT}.")

    results = []
    for i in range(NUM_SIMULATIONS):
        # Use the loop index as a seed to ensure each chaotic path is unique
        result = run_single_simulation(
            num_points=PATH_LENGTH,
            prime_limit=PRIME_SEARCH_LIMIT,
            num_anchors=NUM_ANCHORS,
            seed=i
        )
        if result:
            results.append(result)

        # Print progress
        if (i + 1) % 50 == 0:
            print(f"  ...completed {i + 1}/{NUM_SIMULATIONS} simulations.")

    if not results:
        print("Ultimate Stress Test could not be completed (no valid results).")
    else:
        # --- Analyze the aggregated results ---
        accuracies = [r['accuracy'] for r in results]
        errors = [r['error'] for r in results]

        avg_accuracy = np.mean(accuracies)
        std_dev_accuracy = np.std(accuracies)
        avg_error = np.mean(errors)

        print("\n--- Ultimate Stress Test Analysis ---")
        print(f"Successfully completed {len(results)} simulation runs under chaotic conditions.")
        print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
        print(f"Standard Deviation of Accuracy: {std_dev_accuracy * 100:.2f}%")
        print(f"Average Absolute Error: {avg_error:.4f}")

        if avg_accuracy > 0.85:
            print("\nVERDICT: The model is extraordinarily robust. Its predictive power holds even against a chaotic, non-stationary system.")
        elif avg_accuracy > 0.6:
            print("\nVERDICT: The model shows resilience, but its accuracy is significantly impacted by chaotic conditions.")
        else:
            print("\nVERDICT: The model's predictive power breaks down under chaotic conditions.")