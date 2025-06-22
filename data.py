from sklearn.datasets import load_iris
from torch.utils.data import Dataset
import torch

class IrisDataset(Dataset):
    def __init__(self, device=None, transform=None, target_transform=None):
        # Load the Iris dataset from scikit-learn
        self.data_set = load_iris()
        self.data = torch.from_numpy(self.data_set.data).to(torch.float64)
        self.labels = torch.from_numpy(self.data_set.target).to(torch.long)
        self.device = device if device is not None else torch.device('cpu')
        
        # Move data to device
        self.data = self.data.to(self.device)
        self.labels = self.labels.to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
