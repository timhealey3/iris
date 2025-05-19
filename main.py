import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import IrisDataset
from torch.utils.data import random_split
from linear_model import LinearNeuralNetwork
from non_linear_model import NonLinearNeuralNetwork

class ModelTrainer:
    def __init__(self, model, epochs, learning_rate):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # load data & model
        self.data = IrisDataset(device=self.device)
        self.model = model.to(self.device)

        # split training, testing 70, 30
        self.train_size = int(0.7 * len(self.data))
        self.test_size = len(self.data) - self.train_size

        self.train_dataset, self.test_dataset = random_split(
            self.data, 
            [self.train_size, self.test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(42))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)

        # using cross entropy loss to handle the three possible outputs
        self.loss_fn = nn.CrossEntropyLoss()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def training(self):
        for i in range(self.epochs):
            epoch_loss = 0.0
            n_samples = self.data.__len__()
            for X, y in self.train_dataloader:
                # zero out gradients
                self.optimizer.zero_grad()
                # forward pass
                prediction = self.model(X)
                # Cross Entropy Loss function of predictions and true labels -> also does softmax
                loss = self.loss_fn(prediction, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / n_samples
            print(f"Epoch {i+1} average loss: {avg_loss:.4f}\n")

    def testing(self):
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
    model = NonLinearNeuralNetwork()
    trainer = ModelTrainer(model=model, epochs=100, learning_rate=0.01)
    trainer.training()
    trainer.testing()