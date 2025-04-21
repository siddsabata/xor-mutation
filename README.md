# 🧬 Final Project Plan: Evolving Mutation Rates in a Simple Neural Network on XOR

## 🎯 Research Goal
Investigate the **reduction principle** in evolutionary theory, which predicts that **mutation rates should evolve toward zero** under stable conditions. We aim to test this using a **simple fixed-topology neural network** that learns the **XOR function**, and observe how **mutation rates evolve over time** under **three different selection strategies**.

---

## 🧠 Background
### What is the Reduction Principle?
The reduction principle is a key theoretical result in evolutionary biology. It suggests that when selective pressure is constant and stabilizing (i.e., fitness peaks are fixed and stable), evolution should favor organisms that lower their mutation rates over time. This happens because once a good solution is found, mutations are more likely to degrade it than improve it.

### Why XOR?
The XOR function is a minimal non-linearly separable classification task, requiring a small neural network with at least one hidden layer. It is computationally simple but structurally interesting enough to test whether a population can evolve toward an optimal set of weights and begin suppressing mutations.

---

## 🧪 Experimental Setup

### 1. Neural Network Structure
- Fixed architecture:
  - Input layer: 2 neurons
  - Hidden layer: 2 neurons (e.g., with sigmoid/tanh activation)
  - Output layer: 1 neuron (sigmoid)
- Total number of weights and biases: ~9

### 2. Individual Representation
Each individual in the population will have:
- `weights`: A vector of real-valued weights and biases.
- `sigma`: A real-valued scalar representing the **mutation rate**.

### 3. Initial Conditions
- Randomly initialize:
  - Weights in a small range (e.g., [-1, 1])
  - Mutation rate `sigma` in a range [0.01, 0.5]
- Population size: 50
- Generations: 50 to 100

### 4. Mutation Process
- Weights are mutated using:
  ```
  w_new = w_old + N(0, sigma^2)
  ```
- Mutation rate evolves too:
  ```
  sigma_new = |sigma_old + N(0, sigma_meta^2)|
  ```
  - `sigma_meta` is a fixed hyperparameter (e.g., 0.01)

### 5. Fitness Evaluation
- For each individual:
  - Compute accuracy or mean squared error on all 4 XOR inputs:
    ```
    XOR Truth Table:
    (0, 0) -> 0
    (0, 1) -> 1
    (1, 0) -> 1
    (1, 1) -> 0
    ```
- Normalize fitness scores to be usable by selection functions.

---

## 🔄 Selection Strategies to Compare
We will compare how mutation rates evolve under three different selection rules:

### 1. **Roulette-Wheel Selection**
- Probability of reproduction is proportional to fitness:
  ```
  p_i = fitness_i / sum(fitness)
  ```
- Stochastic: Individuals with higher fitness are more likely to reproduce, but not guaranteed.

### 2. **Tournament Selection**
- Randomly select `k` individuals (e.g., 3).
- The fittest among them is selected to reproduce.
- Adds stochasticity with stronger selection pressure.

### 3. **Deterministic Selection**
- Always select the top `n` individuals with the highest fitness scores to reproduce.
- Fully deterministic: introduces the highest selection pressure.

---

## 📊 Data to Collect
For each selection rule, we will track the following over generations:
- **Best fitness** and **average fitness**
- **Best mutation rate** and **average mutation rate**
- (Optional) Population diversity (e.g., average pairwise distance between weight vectors)

All results will be plotted for comparison:
- Fitness over time
- Mutation rate over time
- Possibly heatmaps or scatterplots of mutation rate vs fitness

---

## 📈 Expected Outcomes
- If the **reduction principle holds**, we expect the **average mutation rate to decrease over time**, especially once the population starts approaching high accuracy on XOR.
- The **rate and extent of reduction** may vary by selection scheme:
  - Deterministic may drive mutation rates down fastest.
  - Tournament may preserve some diversity.
  - Roulette-wheel may retain higher mutation rates longer.

---

## 🧰 Tools
- Python + NumPy (or PyTorch for ease of NN forward passes)
- Matplotlib for plotting
- Seeded randomness for reproducibility

---

## ✅ Summary
This project combines evolutionary theory with a computational model to test a classic prediction — that **mutation rates evolve toward zero under stabilizing selection**. By evolving a simple neural network on a toy XOR task and comparing different selection rules, we can explore how evolutionary dynamics behave in a controlled, analyzable setting.

