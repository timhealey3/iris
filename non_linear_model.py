import torch.nn as nn
import torch

class NonLinearNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Non linear activation
        self.relu = nn.ReLU()
        # generate random weights and biases for the 3 hidden layers
        # Layer 1 input layer
        self.weights = nn.Parameter(torch.rand(4, 16, dtype=torch.float64))
        self.bias = nn.Parameter(torch.rand(16, dtype=torch.float64))
        # Layer 2 hidden layer
        self.weights2 = nn.Parameter(torch.rand(16, 8, dtype=torch.float64))
        self.bias2 = nn.Parameter(torch.rand(8, dtype=torch.float64))
        # Layer 3 output layer
        self.weights3 = nn.Parameter(torch.rand(8, 3, dtype=torch.float64))
        self.bias3 = nn.Parameter(torch.rand(3, dtype=torch.float64))

    # Perform manual matrix multiplication
    # This is less efficient than torch.matmul but I wanted to implement it manually
    def matrixMultiplication(self, x, weights):
        z = torch.zeros(x.shape[0], weights.shape[1], dtype=torch.float64, device=x.device)
        # iterates over each data set in batch
        for i in range(x.shape[0]):
            # iterates over each weight
            for j in range(weights.shape[0]):
                z[i] += x[i][j] * weights[j]
        
        return z

    # Add bias to each data set in batch
    # This is less efficient than python addition but I wanted to implement it manually
    def addBias(self, x, bias):   
        z = torch.zeros(x.shape[0], bias.shape[0], dtype=torch.float64, device=x.device)
        # iterates over each data set in batch
        for i in range(x.shape[0]):
            # iterates over each bias
            for j in range(bias.shape[0]):
                z[i][j] = x[i][j] + bias[j]
        
        return z

    def forward(self, x):
        # Layer 1
        layer_one_x = self.matrixMultiplication(x, self.weights)
        layer_one_z = self.addBias(layer_one_x, self.bias)
        lane_one_activation = self.relu(layer_one_z)
        # Layer 2
        layer_two_x = self.matrixMultiplication(lane_one_activation, self.weights2)
        layer_two_z = self.addBias(layer_two_x, self.bias2)
        layer_two_activation = self.relu(layer_two_z)        
        # Layer 3
        layer_three_x = self.matrixMultiplication(layer_two_activation, self.weights3)
        layer_three_z = self.addBias(layer_three_x, self.bias3)
        return layer_three_z