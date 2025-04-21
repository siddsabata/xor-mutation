"""
Main script for the XOR evolution simulation.
Orchestrates running experiments and plotting results.
"""
import numpy as np
# Import experiment running functions
from experiments import run_experiment_1, run_experiment_2, run_experiment_3, run_experiment_4
# Import plotting functions
from plotting import plot_comparison, plot_scaling_results, plot_scaling_time_series
import os # For checking file existence
import pickle # For saving/loading results objects

# --- Simulation Constants --- 
# Set random seed for reproducibility
SEED = 42

# Population size for fixed/dynamic comparison (Experiments 1 & 2)
# Also included in scaling experiments
POPULATION_SIZE = 1000 
# Number of generations for each evolution run
NUM_GENERATIONS = 1000 
# Number of weights in the neural network
NUM_WEIGHTS = 9 # 2x2 W1 + 2 b1 + 2x1 W2 + 1 b2
# Meta-mutation rate for dynamic sigma strategy
SIGMA_META = 0.01 
# Population sizes for scaling experiments (Experiments 3 & 4)
POPULATION_SIZES_SCALING = [50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000]

# XOR Dataset
XOR_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_OUTPUTS = np.array([0, 1, 1, 0])

# --- Removed function definitions (moved to evolution.py and experiments.py) ---
# initialize_population(...) was moved to evolution.py
# select_parents(...) was moved to evolution.py
# reproduce(...) was moved to evolution.py
# run_evolution(...) was moved to evolution.py
# run_experiment_1(...) was moved to experiments.py
# run_experiment_2(...) was moved to experiments.py
# run_experiment_3(...) was moved to experiments.py
# run_experiment_4(...) was moved to experiments.py

def main():
    print("Starting XOR Evolution Simulation...")

    # Define paths for saving/loading full results
    results_dir = 'data'
    os.makedirs(results_dir, exist_ok=True) # Ensure data directory exists
    exp1_file = os.path.join(results_dir, 'exp1_results.pkl')
    exp2_file = os.path.join(results_dir, 'exp2_results.pkl')
    exp3_file = os.path.join(results_dir, 'exp3_scaling_results.pkl')
    exp4_file = os.path.join(results_dir, 'exp4_scaling_results.pkl')

    # --- Run or Load Experiments --- 
    # Check if results files exist, load if they do, otherwise run
    
    # Experiment 1
    if os.path.exists(exp1_file):
        print(f"Loading results for Experiment 1 from {exp1_file}...")
        with open(exp1_file, 'rb') as f:
            results_exp1 = pickle.load(f)
    else:
        results_exp1 = run_experiment_1(
            pop_size=POPULATION_SIZE,
            num_generations=NUM_GENERATIONS,
            num_weights=NUM_WEIGHTS,
            xor_inputs=XOR_INPUTS,
            xor_outputs=XOR_OUTPUTS,
            sigma_meta=SIGMA_META,
            seed=SEED
        )
        print(f"Saving results for Experiment 1 to {exp1_file}...")
        with open(exp1_file, 'wb') as f:
            pickle.dump(results_exp1, f)

    # Experiment 2
    if os.path.exists(exp2_file):
        print(f"Loading results for Experiment 2 from {exp2_file}...")
        with open(exp2_file, 'rb') as f:
            results_exp2 = pickle.load(f)
    else:
        results_exp2 = run_experiment_2(
            pop_size=POPULATION_SIZE,
            num_generations=NUM_GENERATIONS,
            num_weights=NUM_WEIGHTS,
            xor_inputs=XOR_INPUTS,
            xor_outputs=XOR_OUTPUTS,
            sigma_meta=SIGMA_META,
            seed=SEED
        )
        print(f"Saving results for Experiment 2 to {exp2_file}...")
        with open(exp2_file, 'wb') as f:
            pickle.dump(results_exp2, f)

    # Experiment 3
    if os.path.exists(exp3_file):
        print(f"Loading results for Experiment 3 from {exp3_file}...")
        with open(exp3_file, 'rb') as f:
            results_exp3 = pickle.load(f)
    else:
        results_exp3 = run_experiment_3(
            pop_sizes_scaling=POPULATION_SIZES_SCALING,
            num_generations=NUM_GENERATIONS,
            num_weights=NUM_WEIGHTS,
            xor_inputs=XOR_INPUTS,
            xor_outputs=XOR_OUTPUTS,
            sigma_meta=SIGMA_META,
            seed=SEED
        )
        print(f"Saving results for Experiment 3 to {exp3_file}...")
        with open(exp3_file, 'wb') as f:
            pickle.dump(results_exp3, f)

    # Experiment 4
    if os.path.exists(exp4_file):
        print(f"Loading results for Experiment 4 from {exp4_file}...")
        with open(exp4_file, 'rb') as f:
            results_exp4 = pickle.load(f)
    else:
        results_exp4 = run_experiment_4(
            pop_sizes_scaling=POPULATION_SIZES_SCALING,
            num_generations=NUM_GENERATIONS,
            num_weights=NUM_WEIGHTS,
            xor_inputs=XOR_INPUTS,
            xor_outputs=XOR_OUTPUTS,
            sigma_meta=SIGMA_META,
            seed=SEED
        )
        print(f"Saving results for Experiment 4 to {exp4_file}...")
        with open(exp4_file, 'wb') as f:
            pickle.dump(results_exp4, f)

    # --- Generate Plots --- 
    # Plot comparison for N=1000 runs (Exp 1 vs Exp 2)
    print("\nGenerating comparison plots for Experiments 1 & 2 (N=1000)...")
    plot_comparison(results_exp1, results_exp2, POPULATION_SIZE)

    # Plot final generation metrics vs population size (Exp 3 & 4)
    print("\nGenerating scaling plots for Experiments 3 & 4...")
    plot_scaling_results(results_exp3, results_exp4)
    
    # Plot time series metrics across population sizes (Exp 3 & 4)
    print("\nGenerating scaling time series plots for Experiments 3 & 4...")
    plot_scaling_time_series(results_exp3, results_exp4)

    print("\nSimulation or loading complete.")
    print("Result objects (.pkl) saved/loaded from 'data/' directory.")
    print("Plots generated in 'plots/' directory.")

if __name__ == "__main__":
    main() 