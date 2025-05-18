import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # generate random weights and biases
        # generate 4x3 weights and 3 biases for the 3 possible outputs
        self.weights = nn.Parameter(torch.rand(4, 3, dtype=torch.float64))
        self.bias = nn.Parameter(torch.rand(3, dtype=torch.float64))
        print("weights: ", self.weights)
        print("bias: ", self.bias)

    def forward(self, x):
        z = torch.matmul(x, self.weights)
        z = z + self.bias
        return z