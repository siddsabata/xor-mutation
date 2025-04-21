# 🧪 Implementation Assignment: Simulating Evolving Mutation Rates in a Neural Network Solving XOR

### 📌 Purpose
You are building a simulation to evolve a simple neural network population that learns the XOR function. We will compare two different inheritance strategies for the mutation rate (fixed vs. evolving) and how population size influences the outcome.

This document breaks down all tasks and defines every component mathematically and conceptually so you can implement it from scratch using **NumPy only**.

---

## ✅ Step-by-Step Tasks

---

### **Step 1: Define the XOR Dataset**

The XOR function takes 2 binary inputs and returns 1 if exactly one of them is 1:

| Input (x₁, x₂) | Output |
|----------------|--------|
| (0, 0)         | 0      |
| (0, 1)         | 1      |
| (1, 0)         | 1      |
| (1, 1)         | 0      |

Prepare:
- Input array: shape (4, 2)
- Target output array: shape (4,)

---

### **Step 2: Define the Neural Network**

#### Structure:
- Input layer: 2 nodes
- Hidden layer: 2 nodes (use `tanh` activation)
- Output layer: 1 node (use `sigmoid` activation)

This will give a network with:
- 2×2 weights for input → hidden (`W1`)
- 2 biases for hidden layer (`b1`)
- 2×1 weights for hidden → output (`W2`)
- 1 bias for output layer (`b2`)

#### Total parameters per network:
You'll flatten and concatenate everything into a single vector of 9 parameters:
```
W1 (2x2) + b1 (2) + W2 (2x1) + b2 (1) = 4 + 2 + 2 + 1 = 9 parameters
```

You will write:
- `forward_pass(weights, x)` that takes a 9-length weight vector and applies the forward pass on a single input `x`

---

### **Step 3: Define the Individual**

Each individual in the population will be a Python object or dict with:
- `weights`: a 1D NumPy array of length 9 (neural network parameters)
- `sigma`: a float (the individual's mutation rate)
- `fitness`: the mean squared error (MSE) across the 4 XOR examples (computed later)

---

### **Step 4: Evaluate Fitness**

#### For each individual:
1. Run the forward pass on all 4 XOR inputs.
2. Compute MSE between predictions and true outputs:
\[
\text{MSE} = \frac{1}{4} \sum_{i=1}^4 (y_i^{\text{pred}} - y_i^{\text{true}})^2
\]
3. Since **lower MSE = better**, we define fitness as:
\[
\text{fitness} = 1 - \text{MSE}
\]

Store this value in the individual's record.

---

### **Step 5: Initialize the Population**

For `N` individuals:
- Randomly initialize 9 weights between `[-1, 1]`
- Randomly initialize mutation rate `sigma` between `[0.01, 0.5]`
- Set fitness = 0
- **Set a random seed for NumPy's random number generator for reproducibility.**

Do this for population sizes of:
- 100, 500, 1000, 5000, 10000 (for scaling experiments)

---

### **Step 6: Roulette-Wheel Selection**

This is the reproduction step. Use **roulette-wheel selection** to choose parents:

- For each individual, compute their selection probability as:
\[
p_i = \frac{\text{fitness}_i}{\sum_j \text{fitness}_j}
\]
- Use these probabilities to **randomly sample parents** (with replacement) to form the new generation of `N` offspring.

---

### **Step 7: Reproduction and Mutation**

#### 1. **Fixed Mutation Rate (Experiments 1 & 3)**

- Copy weights from parent
- Add noise to each weight:
\[
w_i^{\text{new}} = w_i^{\text{old}} + \mathcal{N}(0, \sigma^2)
\]
- Use the same `sigma` as the parent (do not change it)

#### 2. **Dynamic Mutation Rate (Experiments 2 & 4)**

- Same weight mutation as above
- Additionally mutate the mutation rate:
\[
\sigma_{\text{new}} = \left| \sigma_{\text{old}} + \mathcal{N}(0, \sigma_{\text{meta}}^2) \right|
\]
Use a small constant like `sigma_meta = 0.01`

---

### **Step 8: Run the Evolution Loop**

For each generation (up to 1000 generations):
1. Evaluate all individuals' fitness
2. Select new parents with roulette-wheel
3. Generate new offspring via mutation
4. Record:
   - Average and best fitness
   - Average and best mutation rate
5. Replace old population with new one

Repeat for:
- Fixed mutation rate (Exp 1)
- Dynamic mutation rate (Exp 2)
- Scaling population sizes (Exps 3 & 4)

---

## 📊 Output to Generate

For each experiment:
- Plot: **Average mutation rate vs. generation**
- Plot: **Best mutation rate vs. generation**
- Plot: **Average fitness vs. generation**

(Optional: Histograms of mutation rate at the end)

---

## 🧠 Recap: What You're Simulating

This project is an evolutionary simulation where each individual is a tiny neural network solving XOR. You will simulate **many generations** of selection and mutation to see whether mutation rates tend to **decrease over time** when allowed to evolve — a prediction from theoretical biology.

---

## 🛠 Tools You Can Use

- Use `np.random.normal(loc=0, scale=sigma)` for Gaussian mutation
- Use `np.random.choice` with weights for roulette-wheel selection
- Use `np.tanh` and `expit` (from `scipy.special` or define your own) for activations
