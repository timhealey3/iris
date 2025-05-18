import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import IrisDataset
from torch.utils.data import random_split
from model import NeuralNetwork

class ModelTrainer:
    def __init__(self):
        # load data & model
        self.data = IrisDataset()
        self.model = NeuralNetwork()

        # split training, testing 80, 20
        self.train_size = int(0.8 * len(self.data))
        self.test_size = len(self.data) - self.train_size

        self.train_dataset, self.test_dataset = random_split(
            self.data, 
            [self.train_size, self.test_size],
            # For reproducibility
            generator=torch.Generator().manual_seed(42)
        )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)

        # using cross entropy loss to handle the three possible outputs
        self.loss_fn = nn.CrossEntropyLoss()

    def training(self):
        epochs = 100
        learning_rate = 0.01
        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        # training loop
        for i in range(epochs):
            epoch_loss = 0.0
            n_samples = self.data.__len__()
            for X, y in self.train_dataloader:
                # zero out gradients
                optimizer.zero_grad()
                # forward pass
                prediction = self.model(X)
                # Cross Entropy Loss function of predictions and true labels -> also does softmax
                loss = self.loss_fn(prediction, y)  # Removed torch.tensor(y) since y is already a tensor
                loss.backward()  # Moved inside the batch loop
                optimizer.step()  # Moved inside the batch loop
                epoch_loss += loss.item()
            avg_loss = epoch_loss / n_samples
            print(f"Epoch {i+1} average loss: {avg_loss:.4f}\n")

    def testing(self):
        # testing loop
        self.model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                prediction = self.model(X)
                test_loss += self.loss_fn(prediction, y).item()
                correct += (torch.argmax(prediction, dim=1) == y).sum().item()
        avg_test_loss = test_loss / len(self.test_dataset)
        accuracy = correct / len(self.test_dataset)
        print(f"Test average loss: {avg_test_loss:.4f}\n")
        print(f"Test accuracy: {accuracy:.4f}\n")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.training()
    trainer.testing()