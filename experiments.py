"""
Functions to run specific simulation experiments.
"""
from evolution import run_evolution

def run_experiment_1(pop_size, num_generations, num_weights, xor_inputs, xor_outputs, sigma_meta, seed):
    """
    Runs Experiment 1: Fixed Mutation Rate.
    Returns results.
    """
    print("\n=== Running Experiment 1: Fixed Mutation Rate ===")
    results = run_evolution(
        pop_size=pop_size, 
        num_generations=num_generations,
        num_weights=num_weights,
        xor_inputs=xor_inputs,
        xor_outputs=xor_outputs,
        strategy='fixed',
        sigma_meta=sigma_meta, 
        seed=seed
    )
    return results

def run_experiment_2(pop_size, num_generations, num_weights, xor_inputs, xor_outputs, sigma_meta, seed):
    """
    Runs Experiment 2: Dynamic Mutation Rate.
    Returns results.
    """
    print("\n=== Running Experiment 2: Dynamic Mutation Rate ===")
    results = run_evolution(
        pop_size=pop_size, 
        num_generations=num_generations,
        num_weights=num_weights,
        xor_inputs=xor_inputs,
        xor_outputs=xor_outputs,
        strategy='dynamic',
        sigma_meta=sigma_meta, 
        seed=seed
    )
    return results

def run_experiment_3(pop_sizes_scaling, num_generations, num_weights, xor_inputs, xor_outputs, sigma_meta, seed):
    """
    Runs Experiment 3: Fixed Mutation Rate across multiple population sizes.
    Returns a dictionary mapping pop_size to results.
    """
    print("\n=== Running Experiment 3: Fixed Mutation Rate (Scaling) ===")
    all_results = {}
    for pop_size in pop_sizes_scaling:
        results = run_evolution(
            pop_size=pop_size, 
            num_generations=num_generations,
            num_weights=num_weights,
            xor_inputs=xor_inputs,
            xor_outputs=xor_outputs,
            strategy='fixed',
            sigma_meta=sigma_meta, 
            seed=seed # Consider adding pop_size to seed if desired: seed=seed + pop_size
        )
        all_results[pop_size] = results
    return all_results

def run_experiment_4(pop_sizes_scaling, num_generations, num_weights, xor_inputs, xor_outputs, sigma_meta, seed):
    """
    Runs Experiment 4: Dynamic Mutation Rate across multiple population sizes.
    Returns a dictionary mapping pop_size to results.
    """
    print("\n=== Running Experiment 4: Dynamic Mutation Rate (Scaling) ===")
    all_results = {}
    for pop_size in pop_sizes_scaling:
        results = run_evolution(
            pop_size=pop_size, 
            num_generations=num_generations,
            num_weights=num_weights,
            xor_inputs=xor_inputs,
            xor_outputs=xor_outputs,
            strategy='dynamic',
            sigma_meta=sigma_meta, 
            seed=seed
        )
        all_results[pop_size] = results
    return all_results 