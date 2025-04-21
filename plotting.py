"""
Plotting functions for the simulation results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_comparison(results_exp1, results_exp2, pop_size, filename_prefix='exp1_exp2'):
    """
    Generates and saves plots comparing results from Experiment 1 (fixed) and Experiment 2 (dynamic).

    Args:
        results_exp1 (tuple): Results tuple from run_evolution (fixed strategy).
        results_exp2 (tuple): Results tuple from run_evolution (dynamic strategy).
        pop_size (int): Population size used.
        filename_prefix (str): Prefix for the output plot filenames.
    """
    
    # Unpack results
    avg_fit1, best_fit1, avg_sig1, best_sig1 = results_exp1
    avg_fit2, best_fit2, avg_sig2, best_sig2 = results_exp2
    
    num_generations = len(avg_fit1) # Should be same for both experiments
    generations = np.arange(1, num_generations + 1)

    # Ensure the plots directory exists
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Fitness vs Generation --- 
    plt.figure(figsize=(12, 6))
    
    # Plot average fitness
    sns.lineplot(x=generations, y=avg_fit1, label='Avg Fitness (Fixed Sigma)', color='#1f77b4')
    sns.lineplot(x=generations, y=avg_fit2, label='Avg Fitness (Dynamic Sigma)', color='#ff7f0e')
    
    # Plot best fitness
    sns.lineplot(x=generations, y=best_fit1, label='Max Fitness (Fixed Sigma)', linestyle='--', color='#1f77b4')
    sns.lineplot(x=generations, y=best_fit2, label='Max Fitness (Dynamic Sigma)', linestyle='--', color='#ff7f0e')

    plt.title(f'Fitness Comparison (N={pop_size})')
    plt.xlabel("Generation")
    plt.ylabel("Fitness (1 - MSE)")
    plt.legend()
    plt.ylim(0, 1.05) # Fitness ranges from 0 to 1
    plt.tight_layout()
    
    # Save fitness plot
    fitness_plot_filename = os.path.join(output_dir, f"{filename_prefix}_fitness_N{pop_size}.png")
    try:
        plt.savefig(fitness_plot_filename)
        print(f"Fitness plot saved to {fitness_plot_filename}")
    except IOError as e:
        print(f"Error saving fitness plot: {e}")
    plt.close() # Close the figure to free memory

    # --- Plot 2: Sigma vs Generation --- 
    plt.figure(figsize=(12, 6))

    # Plot average sigma
    sns.lineplot(x=generations, y=avg_sig1, label='Avg Sigma (Fixed Sigma)', color='#2ca02c')
    sns.lineplot(x=generations, y=avg_sig2, label='Avg Sigma (Dynamic Sigma)', color='#d62728')

    # Plot sigma of the best individual
    sns.lineplot(x=generations, y=best_sig1, label='Max Individual Sigma (Fixed Sigma)', linestyle='--', color='#2ca02c')
    sns.lineplot(x=generations, y=best_sig2, label='Max Individual Sigma (Dynamic Sigma)', linestyle='--', color='#d62728')

    plt.title(f'Mutation Rate (Sigma) Comparison (N={pop_size})')
    plt.xlabel("Generation")
    plt.ylabel("Sigma")
    plt.legend()
    # Optional: Set y-limit if sigmas tend to stay within a range, e.g., plt.ylim(bottom=0)
    plt.tight_layout()

    # Save sigma plot
    sigma_plot_filename = os.path.join(output_dir, f"{filename_prefix}_sigma_N{pop_size}.png")
    try:
        plt.savefig(sigma_plot_filename)
        print(f"Sigma plot saved to {sigma_plot_filename}")
    except IOError as e:
        print(f"Error saving sigma plot: {e}")
    plt.close()

# TODO: Add function for plotting scaling experiment results
def plot_scaling_results(results_exp3, results_exp4, filename_prefix='scaling'):
    """
    Generates and saves plots showing the effect of population size from Experiments 3 & 4.

    Args:
        results_exp3 (dict): Results dictionary from run_experiment_3 (fixed scaling).
                               Maps pop_size -> results tuple.
        results_exp4 (dict): Results dictionary from run_experiment_4 (dynamic scaling).
                               Maps pop_size -> results tuple.
        filename_prefix (str): Prefix for the output plot filenames.
    """
    
    # Extract population sizes tested (should be the same for both experiments)
    pop_sizes = sorted(results_exp3.keys())
    if sorted(results_exp4.keys()) != pop_sizes:
        print("Warning: Population sizes differ between Experiment 3 and 4 results!")
        # Handle potential mismatch if necessary, e.g., use intersection
        pop_sizes = sorted(list(set(results_exp3.keys()) & set(results_exp4.keys())))
        if not pop_sizes:
            print("Error: No common population sizes found for scaling plots.")
            return

    # --- Data Extraction --- 
    # Store final values for each population size
    final_metrics = {'fixed': {'max_fit': [], 'avg_fit': [], 'avg_sig': [], 'max_fit_sig': []},
                     'dynamic': {'max_fit': [], 'avg_fit': [], 'avg_sig': [], 'max_fit_sig': []}}

    for N in pop_sizes:
        # Fixed Strategy (Exp 3)
        if N in results_exp3:
            avg_fit, max_fit, avg_sig, max_fit_sig = results_exp3[N]
            # Get the metric from the last generation (-1 index)
            final_metrics['fixed']['max_fit'].append(max_fit[-1])
            final_metrics['fixed']['avg_fit'].append(avg_fit[-1])
            final_metrics['fixed']['avg_sig'].append(avg_sig[-1])
            final_metrics['fixed']['max_fit_sig'].append(max_fit_sig[-1])
        else: # Handle missing data point if necessary
             final_metrics['fixed']['max_fit'].append(np.nan)
             # ... add nans for other metrics too

        # Dynamic Strategy (Exp 4)
        if N in results_exp4:
            avg_fit, max_fit, avg_sig, max_fit_sig = results_exp4[N]
            final_metrics['dynamic']['max_fit'].append(max_fit[-1])
            final_metrics['dynamic']['avg_fit'].append(avg_fit[-1])
            final_metrics['dynamic']['avg_sig'].append(avg_sig[-1])
            final_metrics['dynamic']['max_fit_sig'].append(max_fit_sig[-1])
        else:
             final_metrics['dynamic']['max_fit'].append(np.nan)
             # ... add nans for other metrics too

    # Ensure the plots directory exists
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Final Fitness vs Population Size --- 
    plt.figure(figsize=(12, 6))
    
    # Max Fitness
    # sns.lineplot(x=pop_sizes, y=final_metrics['fixed']['max_fit'], label='Max Fitness (Fixed Sigma)', marker='o', color='#1f77b4')
    # sns.lineplot(x=pop_sizes, y=final_metrics['dynamic']['max_fit'], label='Max Fitness (Dynamic Sigma)', marker='o', color='#ff7f0e')
    # Average Fitness
    sns.lineplot(x=pop_sizes, y=final_metrics['fixed']['avg_fit'], label='Avg Fitness (Fixed Sigma)', marker='s', linestyle='-', color='#1f77b4')
    sns.lineplot(x=pop_sizes, y=final_metrics['dynamic']['avg_fit'], label='Avg Fitness (Dynamic Sigma)', marker='s', linestyle='-', color='#ff7f0e')

    plt.title('Final Generation Average Fitness vs. Population Size')
    plt.xlabel("Population Size (N)")
    plt.ylabel("Fitness (1 - MSE)")
    plt.xscale('log') # Population size often viewed on log scale
    plt.xticks(pop_sizes, labels=[str(N) for N in pop_sizes]) # Ensure ticks match tested sizes
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    # Save fitness plot
    fitness_plot_filename = os.path.join(output_dir, f"{filename_prefix}_final_fitness.png")
    try:
        plt.savefig(fitness_plot_filename)
        print(f"Scaling fitness plot saved to {fitness_plot_filename}")
    except IOError as e:
        print(f"Error saving scaling fitness plot: {e}")
    plt.close()

    # --- Plot 2: Final Sigma vs Population Size --- 
    plt.figure(figsize=(12, 6))

    # Avg Sigma
    sns.lineplot(x=pop_sizes, y=final_metrics['fixed']['avg_sig'], label='Avg Sigma (Fixed Sigma)', marker='o', color='#2ca02c')
    sns.lineplot(x=pop_sizes, y=final_metrics['dynamic']['avg_sig'], label='Avg Sigma (Dynamic Sigma)', marker='o', color='#d62728')
    # Max Fitness Sigma
    # sns.lineplot(x=pop_sizes, y=final_metrics['fixed']['max_fit_sig'], label='Max Fitness Sigma (Fixed Sigma)', marker='s', linestyle=':', color='#2ca02c')
    # sns.lineplot(x=pop_sizes, y=final_metrics['dynamic']['max_fit_sig'], label='Max Fitness Sigma (Dynamic Sigma)', marker='s', linestyle=':', color='#d62728')

    plt.title('Final Generation Average Sigma vs. Population Size')
    plt.xlabel("Population Size (N)")
    plt.ylabel("Sigma")
    plt.xscale('log')
    plt.xticks(pop_sizes, labels=[str(N) for N in pop_sizes])
    plt.legend()
    plt.tight_layout()

    # Save sigma plot
    sigma_plot_filename = os.path.join(output_dir, f"{filename_prefix}_final_sigma.png")
    try:
        plt.savefig(sigma_plot_filename)
        print(f"Scaling sigma plot saved to {sigma_plot_filename}")
    except IOError as e:
        print(f"Error saving scaling sigma plot: {e}")
    plt.close() 

def plot_scaling_time_series(results_exp3, results_exp4, filename_prefix='scaling_ts'):
    """
    Generates plots showing average fitness and sigma over time across population sizes,
    using stacked subplots for fixed vs dynamic strategies.

    Args:
        results_exp3 (dict): Results dictionary from run_experiment_3 (fixed scaling).
        results_exp4 (dict): Results dictionary from run_experiment_4 (dynamic scaling).
        filename_prefix (str): Prefix for the output plot filenames.
    """
    pop_sizes = sorted(results_exp3.keys())
    if sorted(results_exp4.keys()) != pop_sizes:
        print("Warning: Population sizes differ between Experiment 3 and 4 results!")
        pop_sizes = sorted(list(set(results_exp3.keys()) & set(results_exp4.keys())))
        if not pop_sizes:
            print("Error: No common population sizes found for scaling time series plots.")
            return

    # Assume all runs have the same number of generations
    # Get generation count from the first result set available
    first_N = pop_sizes[0]
    if first_N in results_exp3:
        num_generations = len(results_exp3[first_N][0]) # Length of avg_fitness list
    elif first_N in results_exp4:
        num_generations = len(results_exp4[first_N][0])
    else:
        print("Error: Cannot determine number of generations.")
        return 
    generations = np.arange(1, num_generations + 1)

    # Ensure the plots directory exists
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    # Set seaborn style and color palette
    sns.set_theme(style="whitegrid")
    # Use a palette that works well for multiple lines
    palette = sns.color_palette("viridis", n_colors=len(pop_sizes))

    # --- Plot 1: Average Fitness vs Generation (Scaling) --- 
    # Create figure with 2 vertically stacked subplots, sharing the Y axis
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    fig.suptitle('Average Fitness vs. Generation (Scaling Comparison)') # Main title

    # Determine overall y-limits (optional, sharey=True usually handles this well)
    # min_overall_avg_fitness = 1.0 
    # max_overall_avg_fitness = 0
    # ... loop through all data first to find min/max ...

    for i, N in enumerate(pop_sizes):
        color = palette[i]
        label_fixed = f'N={N}'
        label_dynamic = f'N={N}'
        
        # Top Subplot: Fixed Strategy (Exp 3)
        if N in results_exp3:
            avg_fit, _, _, _ = results_exp3[N]
            sns.lineplot(x=generations, y=avg_fit, label=label_fixed, color=color, linestyle='-', ax=axes[0])
            # min_overall_avg_fitness = min(min_overall_avg_fitness, np.min(avg_fit))
            # max_overall_avg_fitness = max(max_overall_avg_fitness, np.max(avg_fit))
            
        # Bottom Subplot: Dynamic Strategy (Exp 4)
        if N in results_exp4:
            avg_fit, _, _, _ = results_exp4[N]
            sns.lineplot(x=generations, y=avg_fit, label=label_dynamic, color=color, linestyle='-', ax=axes[1])
            # min_overall_avg_fitness = min(min_overall_avg_fitness, np.min(avg_fit))
            # max_overall_avg_fitness = max(max_overall_avg_fitness, np.max(avg_fit))

    # --- Finalize Fitness Plot ---
    axes[0].set_title('Fixed Sigma Strategy')
    axes[0].legend(title="Pop Size", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_ylabel("Average Fitness (1 - MSE)")
    axes[0].tick_params(labelbottom=False) # Hide x-axis labels for top plot

    axes[1].set_title('Dynamic Sigma Strategy')
    axes[1].legend(title="Pop Size", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Average Fitness (1 - MSE)")

    # axes[0].set_ylim(min_overall_avg_fitness - 0.05, max_overall_avg_fitness + 0.05) # Manual ylim if needed
    # Set limit slightly above 1.0 if fitness reaches it
    current_ymin, current_ymax = axes[1].get_ylim() # Get ylim set by sharey
    axes[1].set_ylim(bottom=max(0, current_ymin), top=min(current_ymax, 1.05)) # Ensure ymin>=0, ymax<=1.05

    plt.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust layout for main title and external legends
    
    # Save fitness plot
    fitness_plot_filename = os.path.join(output_dir, f"{filename_prefix}_avg_fitness.png")
    try:
        plt.savefig(fitness_plot_filename)
        print(f"Scaling time series fitness plot saved to {fitness_plot_filename}")
    except IOError as e:
        print(f"Error saving scaling time series fitness plot: {e}")
    plt.close()

    # --- Plot 2: Average Sigma vs Generation (Scaling) --- 
    # Create figure with 2 vertically stacked subplots, sharing the Y axis
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    fig.suptitle('Average Sigma vs. Generation (Scaling Comparison)') # Main title
    
    # Determine overall y-limits (optional, sharey=True usually handles this well)
    # min_overall_avg_sigma = float('inf')
    # max_overall_avg_sigma = 0
    # ... loop through all data first to find min/max ...

    for i, N in enumerate(pop_sizes):
        color = palette[i]
        label_fixed = f'N={N}'
        label_dynamic = f'N={N}'

        # Top Subplot: Fixed Strategy (Exp 3)
        if N in results_exp3:
            _, _, avg_sig, _ = results_exp3[N]
            sns.lineplot(x=generations, y=avg_sig, label=label_fixed, color=color, linestyle='-', ax=axes[0])
            # min_overall_avg_sigma = min(min_overall_avg_sigma, np.min(avg_sig))
            # max_overall_avg_sigma = max(max_overall_avg_sigma, np.max(avg_sig))
            
        # Bottom Subplot: Dynamic Strategy (Exp 4)
        if N in results_exp4:
             _, _, avg_sig, _ = results_exp4[N]
             sns.lineplot(x=generations, y=avg_sig, label=label_dynamic, color=color, linestyle='-', ax=axes[1])
             # min_overall_avg_sigma = min(min_overall_avg_sigma, np.min(avg_sig))
             # max_overall_avg_sigma = max(max_overall_avg_sigma, np.max(avg_sig))

    # --- Finalize Sigma Plot --- 
    axes[0].set_title('Fixed Sigma Strategy')
    axes[0].legend(title="Pop Size", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].set_ylabel("Average Sigma")
    axes[0].tick_params(labelbottom=False) # Hide x-axis labels for top plot

    axes[1].set_title('Dynamic Sigma Strategy')
    axes[1].legend(title="Pop Size", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Average Sigma")

    # axes[0].set_ylim(bottom=max(0, min_overall_avg_sigma - 0.02), top=max_overall_avg_sigma + 0.02) # Manual ylim
    current_ymin, current_ymax = axes[1].get_ylim() # Get ylim set by sharey
    axes[1].set_ylim(bottom=max(0, current_ymin)) # Ensure ymin>=0

    plt.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust layout for main title and external legends

    # Save sigma plot
    sigma_plot_filename = os.path.join(output_dir, f"{filename_prefix}_avg_sigma.png")
    try:
        plt.savefig(sigma_plot_filename)
        print(f"Scaling time series sigma plot saved to {sigma_plot_filename}")
    except IOError as e:
        print(f"Error saving scaling time series sigma plot: {e}")
    plt.close() 