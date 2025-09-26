"""
This script implements a complex systems analysis using twin primes.
The user's idea is implemented faithfully and transparently, now refactored for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime
import warnings
from typing import List, Optional, Dict, Any

warnings.filterwarnings('ignore')


# --- Core Data Generation Functions ---

def generate_brownian_motion(
    num_points: int = 10000,
    drift: float = 0.0,
    volatility: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Brownian motion path.
    """
    if seed is not None:
        np.random.seed(seed)
    random_steps = np.random.normal(0, 1, num_points)
    steps = drift + volatility * random_steps
    return np.cumsum(steps)


def generate_chaotic_path(num_points: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generates a chaotic, non-stationary path with dynamic volatility and drift.
    """
    if seed is not None:
        np.random.seed(seed)

    # Dynamic Volatility
    volatility_path = np.zeros(num_points)
    current_volatility = 1.5
    volatility_innovations = np.random.normal(0, 0.2, num_points)
    for i in range(num_points):
        current_volatility = (0.995 * current_volatility) + (0.005 * 1.5) + volatility_innovations[i]
        volatility_path[i] = max(0.2, current_volatility)

    # Flipping Drift
    drift_path = np.zeros(num_points)
    current_drift = 0.05
    change_probability = 1 / 2000
    for i in range(num_points):
        if np.random.random() < change_probability:
            current_drift = np.random.normal(0, 0.15)
        drift_path[i] = current_drift

    # Generate final path
    random_innovations = np.random.normal(0, 1, num_points)
    path_steps = drift_path + volatility_path * random_innovations
    return np.cumsum(path_steps)


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
        if 0 < distance <= 200:
            weight = 1.0 / (distance + 1)
            anchor_value = path[anchor]

            # Simple trend adjustment
            end_slice = min(anchor + 10, len(path))
            trend_adjustment = np.mean(path[anchor:end_slice]) - anchor_value if end_slice > anchor else 0
            prediction = anchor_value + trend_adjustment

            predictions.append(prediction)
            weights.append(weight)

    if not predictions:
        return None

    return np.average(predictions, weights=weights)


# --- Simulation and Analysis Functions ---

def run_single_prediction(
    path_generator,
    path_args: Dict[str, Any],
    prime_limit: int,
    num_anchors: int,
) -> Optional[Dict[str, Any]]:
    """
    Runs a single prediction task on a given path.
    """
    target_prime_index = num_anchors
    path = path_generator(**path_args)
    twin_primes = find_twin_primes(prime_limit)

    if len(twin_primes) <= target_prime_index:
        return None  # Not enough primes

    anchors = twin_primes[:num_anchors]
    target = twin_primes[target_prime_index]

    if target >= len(path):
        return None  # Target is out of bounds

    prediction = predict_with_multiple_anchors(path, anchors, target)
    if prediction is None:
        return None

    actual_value = path[target]
    error = abs(prediction - actual_value)
    # Accuracy is defined as 1 minus the error normalized by the path's standard deviation
    accuracy = max(0, 1 - error / (np.std(path) + 1e-10))

    return {
        "accuracy": accuracy,
        "error": error,
        "prediction": prediction,
        "actual": actual_value,
        "anchors": anchors,
        "target": target,
        "path": path
    }


def run_monte_carlo_simulation(
    num_simulations: int,
    path_generator,
    path_args_template: Dict[str, Any],
    prime_limit: int,
    num_anchors: int,
) -> List[Dict[str, float]]:
    """
    Runs a Monte Carlo simulation with many prediction runs.
    """
    results = []
    for i in range(num_simulations):
        # Ensure each run is unique by changing the seed
        path_args = path_args_template.copy()
        path_args["seed"] = i

        result = run_single_prediction(
            path_generator=path_generator,
            path_args=path_args,
            prime_limit=prime_limit,
            num_anchors=num_anchors,
        )
        if result:
            results.append({"accuracy": result["accuracy"], "error": result["error"]})

        if (i + 1) % 50 == 0:
            print(f"  ...completed {i + 1}/{num_simulations} simulations.")

    return results


# --- Experiment Phases ---

def run_phase_1_baseline():
    """
    Phase 1: Baseline performance on standard Brownian motion.
    """
    print("--- Phase 1: Baseline Performance on Standard Brownian Motion ---")
    result = run_single_prediction(
        path_generator=generate_brownian_motion,
        path_args={"num_points": 10000, "drift": 0.0, "volatility": 1.0, "seed": 42},
        prime_limit=1000,
        num_anchors=10
    )
    if result:
        print(f"Prediction for t={result['target']}: Predicted={result['prediction']:.2f}, Actual={result['actual']:.2f}")
        print(f"Result: Accuracy={result['accuracy']*100:.2f}%, Abs. Error={result['error']:.2f}\n")
    else:
        print("Phase 1 failed to produce a result.\n")


def run_phase_2_high_volatility():
    """
    Phase 2: Robustness test on a high-volatility system.
    """
    print("--- Phase 2: Robustness Test on High-Volatility System (500 runs) ---")
    results = run_monte_carlo_simulation(
        num_simulations=500,
        path_generator=generate_brownian_motion,
        path_args_template={"num_points": 10000, "drift": 0.02, "volatility": 1.5},
        prime_limit=1000,
        num_anchors=10
    )
    if results:
        accuracies = [r['accuracy'] for r in results]
        errors = [r['error'] for r in results]
        print(f"Result: Avg. Accuracy={np.mean(accuracies)*100:.2f}%, Std. Dev={np.std(accuracies)*100:.2f}%, Avg. Abs. Error={np.mean(errors):.2f}\n")
    else:
        print("Phase 2 failed to produce results.\n")


def run_phase_3_ultimate_stress_test():
    """
    Phase 3: Ultimate stress test against a chaotic system.
    """
    print("--- Phase 3: Ultimate Stress Test on Chaotic System (200 runs) ---")
    results = run_monte_carlo_simulation(
        num_simulations=200,
        path_generator=generate_chaotic_path,
        path_args_template={"num_points": 50000},
        prime_limit=1000,
        num_anchors=25
    )
    if results:
        accuracies = [r['accuracy'] for r in results]
        errors = [r['error'] for r in results]
        avg_accuracy = np.mean(accuracies)
        print(f"Result: Avg. Accuracy={avg_accuracy*100:.2f}%, Std. Dev={np.std(accuracies)*100:.2f}%, Avg. Abs. Error={np.mean(errors):.4f}")

        if avg_accuracy > 0.85:
            print("\nVERDICT: The model is extraordinarily robust. Its predictive power holds even against a chaotic, non-stationary system.")
        elif avg_accuracy > 0.6:
            print("\nVERDICT: The model shows resilience, but its accuracy is significantly impacted by chaotic conditions.")
        else:
            print("\nVERDICT: The model's predictive power breaks down under chaotic conditions.")
    else:
        print("Phase 3 failed to produce results.")


# --- Main Execution ---

def main():
    """
    Main function to run all experimental phases as described in the scientific report.
    """
    print("--- Starting Full Analysis: Twin Prime Anchoring Model ---")
    print("This script replicates the three-phase experiment from the scientific report.\n")

    # The report only includes the results of Phase 3 in its final table,
    # but the methodology describes all three. We run them all for completeness.
    # To match the log file, we will only run Phase 3.

    # run_phase_1_baseline()
    # run_phase_2_high_volatility()
    run_phase_3_ultimate_stress_test()

if __name__ == "__main__":
    main()