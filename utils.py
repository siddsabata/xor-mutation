"""
Utility functions for the simulation.
"""
import numpy as np
# Need access to the forward_pass function to evaluate the network
from network import forward_pass 
import csv # For saving results
import os # For creating directories

# The sigmoid function was moved to network.py to avoid circular imports
# def sigmoid(x):
#     """Compute the sigmoid activation function."""
#     # Clamp x to avoid overflow in exp
#     x = np.clip(x, -500, 500)
#     return 1 / (1 + np.exp(-x))

# TODO: Add fitness calculation function 
def calculate_fitness(weights_flat, xor_inputs, xor_outputs):
    """
    Calculates the fitness of an individual based on its performance on the XOR task.

    Args:
        weights_flat (np.ndarray): The flattened weights (9,) of the neural network.
        xor_inputs (np.ndarray): The XOR input patterns (4, 2).
        xor_outputs (np.ndarray): The target XOR output patterns (4,).

    Returns:
        float: The fitness score (1 - MSE).
    """
    predictions = []
    # Iterate through each XOR input pattern
    for x_input in xor_inputs:
        # Get the network's prediction for the current input
        prediction = forward_pass(weights_flat, x_input)
        predictions.append(prediction)
    
    # Convert predictions to a NumPy array for calculation
    predictions = np.array(predictions)
    
    # Calculate Mean Squared Error (MSE)
    # Ensure outputs are compared correctly (both should be flat arrays)
    mse = np.mean((predictions - xor_outputs)**2)
    
    # Fitness is defined as 1 - MSE (higher is better)
    fitness = 1.0 - mse
    
    return fitness 

def save_results_to_csv(results, strategy, pop_size, filename_prefix='results'):
    """
    Saves the simulation results history to a CSV file.

    Args:
        results (tuple): The tuple returned by run_evolution containing history lists.
        strategy (str): The mutation strategy ('fixed' or 'dynamic').
        pop_size (int): The population size used in the simulation.
        filename_prefix (str): Prefix for the output filename.
    """
    avg_fitness_history, best_fitness_history, avg_sigma_history, best_sigma_history = results
    num_generations = len(avg_fitness_history)
    
    # Ensure the data directory exists
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    # Construct filename
    filename = os.path.join(output_dir, f"{filename_prefix}_{strategy}_N{pop_size}.csv")
    print(f"Saving results to {filename}...")

    # Prepare data for CSV writing (list of rows)
    header = ['Generation', 'AvgFitness', 'MaxFitness', 'AvgSigma', 'MaxFitnessSigma']
    rows = [header]
    for gen in range(num_generations):
        row = [
            gen + 1, # Generation number (1-indexed)
            avg_fitness_history[gen],
            best_fitness_history[gen],
            avg_sigma_history[gen],
            best_sigma_history[gen]
        ]
        rows.append(row)

    # Write to CSV
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        print(f"Results saved successfully.")
    except IOError as e:
        print(f"Error saving results to {filename}: {e}") 