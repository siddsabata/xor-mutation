"""
Core evolutionary algorithm functions.
"""
import numpy as np
import time
from utils import calculate_fitness

# --- Population Initialization --- 
def initialize_population(pop_size, num_weights):
    """
    Initializes the population with random weights and sigma values.
    Args:
        pop_size (int): The number of individuals in the population.
        num_weights (int): The number of weights in each individual's network.
    Returns:
        list: A list of dictionaries, where each dictionary represents an individual.
    """
    population = []
    for _ in range(pop_size):
        weights = np.random.uniform(-1.0, 1.0, size=num_weights)
        sigma = np.random.uniform(0.01, 0.5)
        individual = {
            'weights': weights,
            'sigma': sigma,
            'fitness': 0.0 
        }
        population.append(individual)
    return population

# --- Selection --- 
def select_parents(population):
    """
    Selects parents from the population using roulette-wheel selection.
    Args:
        population (list): The current population of individuals.
    Returns:
        list: A list of selected parent individuals (references to originals).
    """
    fitness_values = np.array([ind['fitness'] for ind in population])
    min_fitness = np.min(fitness_values)
    adjusted_fitness = fitness_values
    if min_fitness < 0:
        adjusted_fitness = fitness_values - min_fitness 
    total_adjusted_fitness = np.sum(adjusted_fitness)
    if total_adjusted_fitness == 0:
        probabilities = np.ones(len(population)) / len(population)
    else:
        probabilities = adjusted_fitness / total_adjusted_fitness
    # Ensure probabilities sum to 1 (due to potential floating point inaccuracies)
    probabilities /= np.sum(probabilities)
    num_parents = len(population)
    parent_indices = np.random.choice(
        a=len(population), size=num_parents, replace=True, p=probabilities
    )
    selected_parents = [population[i] for i in parent_indices]
    return selected_parents

# --- Reproduction --- 
def reproduce(parents, strategy, num_weights, sigma_meta):
    """
    Creates the next generation of offspring through mutation.
    Args:
        parents (list): List of selected parent individuals.
        strategy (str): The mutation strategy ('fixed' or 'dynamic').
        num_weights (int): Number of weights per individual.
        sigma_meta (float): The meta-mutation rate for sigma (used in 'dynamic').
    Returns:
        list: The new offspring population.
    """
    offspring_population = []
    for parent in parents:
        offspring = {}
        offspring_sigma = parent['sigma']
        noise = np.random.normal(loc=0.0, scale=offspring_sigma, size=num_weights)
        offspring_weights = parent['weights'] + noise
        if strategy == 'dynamic':
            sigma_noise = np.random.normal(loc=0.0, scale=sigma_meta)
            offspring_sigma = abs(offspring_sigma + sigma_noise)
            # Optional: Add a small minimum value to prevent sigma from becoming exactly zero
            offspring_sigma = max(offspring_sigma, 1e-6) 
        offspring['weights'] = offspring_weights
        offspring['sigma'] = offspring_sigma
        offspring['fitness'] = 0.0 
        offspring_population.append(offspring)
    return offspring_population

# --- Main Evolution Loop --- 
def run_evolution(
    pop_size, num_generations, num_weights, xor_inputs, xor_outputs, 
    strategy, sigma_meta, seed
):
    """
    Runs the evolutionary simulation for a given strategy.
    Args:
        pop_size (int): Population size.
        num_generations (int): Number of generations to run.
        num_weights (int): Number of weights per network.
        xor_inputs (np.ndarray): XOR input data.
        xor_outputs (np.ndarray): XOR target data.
        strategy (str): Mutation strategy ('fixed' or 'dynamic').
        sigma_meta (float): Meta-mutation rate for sigma (dynamic strategy).
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: Contains lists of recorded metrics over generations:
               (avg_fitness_history, max_fitness_history, 
                avg_sigma_history, max_fitness_sigma_history)
    """
    np.random.seed(seed)
    print(f"\n--- Running Evolution: Strategy='{strategy}', Population Size={pop_size}, Generations={num_generations} ---")
    start_time = time.time()
    # Initialize population using the function defined in this file
    population = initialize_population(pop_size, num_weights)
    
    # Data recorders
    avg_fitness_history = []
    max_fitness_history = []
    avg_sigma_history = []
    max_fitness_sigma_history = []
    
    # Evolution Loop
    for generation in range(num_generations):
        fitness_scores = []
        sigmas = []
        max_fitness_current_gen = -np.inf
        max_fitness_individual_sigma = -1.0
        
        # Evaluate fitness for all individuals
        for i in range(len(population)):
            fitness = calculate_fitness(population[i]['weights'], xor_inputs, xor_outputs)
            population[i]['fitness'] = fitness
            fitness_scores.append(fitness)
            sigmas.append(population[i]['sigma'])
            
            # Track the individual with max fitness in this generation
            if fitness > max_fitness_current_gen:
                max_fitness_current_gen = fitness
                max_fitness_individual_sigma = population[i]['sigma']
        
        # Record average and max metrics for the generation
        avg_fitness = np.mean(fitness_scores)
        avg_sigma = np.mean(sigmas)
        avg_fitness_history.append(avg_fitness)
        max_fitness_history.append(max_fitness_current_gen)
        avg_sigma_history.append(avg_sigma)
        max_fitness_sigma_history.append(max_fitness_individual_sigma)
        
        # Print progress periodically
        if (generation + 1) % 100 == 0:
            print(f"Generation {generation + 1}/{num_generations} | "
                  f"Avg Fitness: {avg_fitness:.4f} | Max Fitness: {max_fitness_current_gen:.4f} | "
                  f"Avg Sigma: {avg_sigma:.4f} | Max Fitness Sigma: {max_fitness_individual_sigma:.4f}")
            
        # Select parents for the next generation
        parents = select_parents(population)
        
        # Create offspring through reproduction and mutation
        offspring = reproduce(parents, strategy, num_weights, sigma_meta)
        
        # Replace the old population with the new offspring
        population = offspring
        
    end_time = time.time()
    print(f"Evolution finished for strategy '{strategy}'. Time taken: {end_time - start_time:.2f} seconds.")
    
    # Return recorded history
    return avg_fitness_history, max_fitness_history, avg_sigma_history, max_fitness_sigma_history 