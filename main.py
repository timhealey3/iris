import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import IrisDataset


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

data = IrisDataset()
model = NeuralNetwork()
train_dataloader = DataLoader(data, batch_size=16, shuffle=True)
epochs = 100
# optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_fn = nn.CrossEntropyLoss()
for i in range(epochs):
    epoch_loss = 0.0
    n_samples = data.__len__()
    for X, y in train_dataloader:
        optimizer.zero_grad()
        # forward pass
        prediction = model(X)
        # Cross Entropy Loss -> also does softmax
        loss = loss_fn(prediction, torch.tensor(y))
        epoch_loss += loss.item()
        # update gradients
        loss.backward()
        # apply gradients
        optimizer.step()
    avg_loss = epoch_loss / n_samples
    print(f"Epoch {i+1} average loss: {avg_loss:.4f}\n")