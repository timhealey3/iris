from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

class IrisDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        # Load the Iris dataset from scikit-learn
        self.data_set = load_iris()
        self.data = torch.from_numpy(self.data_set.data)
        self.labels = torch.from_numpy(self.data_set.target)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_idx = self.data[idx]
        label_idx = self.labels[idx]
        return data_idx, label_idx
