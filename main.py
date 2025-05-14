import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import IrisDataset
from torch.utils.data import random_split

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

# load data & model
data = IrisDataset()
model = NeuralNetwork()

# split training, testing 80, 20
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

train_dataset, test_dataset = random_split(
    data, 
    [train_size, test_size],
    # For reproducibility
    generator=torch.Generator().manual_seed(42)
)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

epochs = 100
learning_rate = 0.01
# optimizer
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
# using cross entropy loss to handle the three possible outputs
loss_fn = nn.CrossEntropyLoss()
# training loop
for i in range(epochs):
    epoch_loss = 0.0
    n_samples = data.__len__()
    for X, y in train_dataloader:
        # zero out gradients
        optimizer.zero_grad()
        # forward pass
        prediction = model(X)
        # Cross Entropy Loss function of predictions and true labels -> also does softmax
        loss = loss_fn(prediction, torch.tensor(y))
        epoch_loss += loss.item()
        # update gradients
        loss.backward()
        # apply gradients
        optimizer.step()
    avg_loss = epoch_loss / n_samples
    print(f"Epoch {i+1} average loss: {avg_loss:.4f}\n")

# testing loop
model.eval()
test_loss = 0.0
correct = 0
with torch.no_grad():
    for X, y in test_dataloader:
        prediction = model(X)
        test_loss += loss_fn(prediction, torch.tensor(y)).item()
        correct += (torch.argmax(prediction, dim=1) == torch.tensor(y)).sum().item()
avg_test_loss = test_loss / len(test_dataset)
accuracy = correct / len(test_dataset)
print(f"Test average loss: {avg_test_loss:.4f}\n")
print(f"Test accuracy: {accuracy:.4f}\n")
