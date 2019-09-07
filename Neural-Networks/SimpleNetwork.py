import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1 - x)


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random Starting Synaptic Weights.")
print(synaptic_weights)

for t in range(60000):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # Calculate Errors.
    error = training_outputs - outputs
    # Make Adjustments.
    adjustments = error * sigmoidDerivative(outputs)
    # Update the Weights.
    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Outputs After Training.")
print(outputs)
