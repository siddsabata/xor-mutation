"""
Neural network definition and forward pass.
"""
import numpy as np
# from utils import sigmoid # No longer needed, sigmoid defined locally

def sigmoid(x):
    """Compute the sigmoid activation function."""
    # Clamp x to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def forward_pass(weights_flat, x):
    """
    Performs the forward pass of the 2-2-1 neural network.

    Args:
        weights_flat (np.ndarray): A 1D NumPy array of 9 weights.
        x (np.ndarray): A 1D NumPy array representing a single input (length 2).

    Returns:
        float: The output of the network for the given input.
    """
    if weights_flat.shape[0] != 9:
        raise ValueError(f"Expected 9 weights, got {weights_flat.shape[0]}")
    if x.shape[0] != 2:
        raise ValueError(f"Expected input of length 2, got {x.shape[0]}")

    # --- Reshape weights --- 
    # W1: input (2) -> hidden (2) => 2x2 = 4 weights
    # b1: hidden (2) biases       => 2 weights
    # W2: hidden (2) -> output (1) => 2x1 = 2 weights
    # b2: output (1) bias        => 1 weight
    # Total = 4 + 2 + 2 + 1 = 9

    W1 = weights_flat[0:4].reshape((2, 2))
    b1 = weights_flat[4:6] 
    W2 = weights_flat[6:8].reshape((2, 1))
    b2 = weights_flat[8]

    # --- Forward Propagation --- 
    # Input to Hidden
    # z1 shape: (1, 2) = (1, 2) @ (2, 2) + (1, 2)
    z1 = x @ W1 + b1
    # Hidden activation (tanh)
    # a1 shape: (1, 2)
    a1 = np.tanh(z1)

    # Hidden to Output
    # z2 shape: (1, 1) = (1, 2) @ (2, 1) + (1,)
    z2 = a1 @ W2 + b2
    # Output activation (sigmoid)
    # output shape: (1, 1) -> scalar
    output = sigmoid(z2).item() # .item() extracts the scalar value

    return output 