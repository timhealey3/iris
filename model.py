import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # generate random weights and biases
        # generate 4x3 weights and 3 biases for the 3 possible outputs
        self.weights = nn.Parameter(torch.rand(4, 3, dtype=torch.float64))
        self.bias = nn.Parameter(torch.rand(3, dtype=torch.float64))

    # Perform manual matrix multiplication
    # This is less efficient than torch.matmul but I wanted to implement it manually
    def matrixMultiplication(self, x):
        z = torch.zeros(x.shape[0], self.weights.shape[1], dtype=torch.float64, device=x.device)
        # iterates over each data set in batch
        for i in range(x.shape[0]):
            # iterates over each weight
            for j in range(self.weights.shape[0]): 
                z[i] += x[i][j] * self.weights[j]
        
        return z

    # Add bias to each data set in batch
    # This is less efficient than python addition but I wanted to implement it manually
    def addBias(self, x):
        z = torch.zeros(x.shape[0], self.bias.shape[0], dtype=torch.float64, device=x.device)
        # iterates over each data set in batch
        for i in range(x.shape[0]):
            # iterates over each bias
            for j in range(self.bias.shape[0]):
                z[i][j] = x[i][j] + self.bias[j]
        
        return z

    def forward(self, x):
        z = self.matrixMultiplication(x)
        z = self.addBias(z)
        return z