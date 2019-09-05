import numpy as np
import pandas as pd


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def train(self, trainingInputs, trainingOutputs, networkIterations):

        for t in range(networkIterations):
            outputs = self.think(trainingInputs)
            error = trainingOutputs - outputs
            adjustments = np.dot(trainingInputs.T, error *
                                 self.sigmoidDerivative(outputs))

            self.synaptic_weights += adjustments

        return outputs

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":
    neural_net = NeuralNetwork()

    print("Random Synaptic Weights : ")
    print(neural_net.synaptic_weights)

    trainingInputs = np.array([[0, 0, 1],
                               [1, 1, 1],
                               [1, 0, 1],
                               [0, 1, 1],
                               [0, 1, 0]])

    trainingOutputs = np.array([[0, 1, 1, 0, 1]]).T

    neural_net.train(trainingInputs, trainingOutputs, 5000)

    print("Synaptic Weights after Training : ")
    print(neural_net.synaptic_weights)

    X1 = str(input("Input 1 : "))
    X2 = str(input("Input 2 : "))
    X3 = str(input("Input 3 : "))

    print("Output Data : ")
    print(neural_net.think(np.array([X1, X2, X3])))
