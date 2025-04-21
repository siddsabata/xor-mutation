# 🧠 XOR Evolution Simulation

This project simulates the evolution of simple neural networks learning the XOR function. It specifically investigates how the strategy for inheriting mutation rates affects the evolutionary process.

## Purpose

The main goal is to compare two scenarios:
1.  **Fixed Mutation Rate:** Each network inherits its mutation strength (`sigma`) directly from its parent without change.
2.  **Dynamic Mutation Rate:** The mutation strength (`sigma`) itself is also subject to mutation and can evolve over generations.

We simulate this to see if mutation rates tend to decrease over time when allowed to evolve, as predicted by some theories.

## How it Works

-   **Population:** We maintain a population of individuals, where each individual represents a small neural network (2 input, 2 hidden, 1 output neurons).
-   **Fitness:** Each network's fitness is determined by how well it solves the XOR problem (calculated as 1 - Mean Squared Error).
-   **Selection:** Individuals with higher fitness have a higher chance of being selected as parents for the next generation (using Roulette Wheel Selection).
-   **Reproduction & Mutation:** Selected parents produce offspring. Offspring inherit the parent's network weights with added random noise (mutation). The strength of this noise is determined by the individual's `sigma`. In the dynamic strategy, the `sigma` value itself is also mutated.
-   **Generations:** This cycle repeats for many generations (e.g., 1000).

## Experiments Conducted

The simulation runs the following experiments:
1.  **Experiment 1:** Fixed Sigma strategy with Population Size N=1000.
2.  **Experiment 2:** Dynamic Sigma strategy with Population Size N=1000.
3.  **Experiment 3:** Fixed Sigma strategy, testing multiple population sizes (scaling analysis).
4.  **Experiment 4:** Dynamic Sigma strategy, testing multiple population sizes (scaling analysis).

## Code Structure

-   `main.py`: Main script to run the simulation. Orchestrates experiments and plotting.
-   `evolution.py`: Contains the core functions of the evolutionary algorithm (initialization, selection, reproduction, evolution loop).
-   `experiments.py`: Defines functions to run each specific experiment.
-   `network.py`: Defines the neural network structure and the `forward_pass` function.
-   `utils.py`: Helper functions (e.g., `calculate_fitness`, `save_results_to_csv` - *Note: CSV saving is currently replaced by pickle saving*).
-   `plotting.py`: Functions to generate plots from the simulation results.
-   `requirements.txt`: Lists Python dependencies.
-   `data/`: Directory where simulation results (`.pkl` files) are saved/loaded.
-   `plots/`: Directory where output plots (`.png`) are saved.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Simulation:**
    ```bash
    python main.py
    ```

3.  **Results Persistence:**
    -   The script saves the complete results of each experiment (or set of scaling experiments) to `.pkl` files in the `data/` directory (e.g., `data/exp1_results.pkl`, `data/exp3_scaling_results.pkl`).
    -   On subsequent runs, the script will **load** these files if they exist, skipping the potentially long simulation runs. This allows for quick regeneration of plots.
    -   To **force a rerun** of the experiments, delete the corresponding `.pkl` files from the `data/` directory.

## Output

-   **Data:** Python object files (`.pkl`) containing the full results history for each experiment are saved in the `data/` directory.
-   **Plots:** Several plots comparing fitness and sigma values (over generations and against population size) are saved as `.png` files in the `plots/` directory. 