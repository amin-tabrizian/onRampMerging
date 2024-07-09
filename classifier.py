import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Create a sample dataset (you can replace this with your actual data)
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define a CNN model with a fully connected layer for classification
class CNNClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_size)  # Adjust the input size based on your data

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define hyperparameters
input_size = 93 # Input size: 31 sequences of length 3
output_size = 31  # Number of classes


# Split data into training and testing sets

# Initialize the model and optimizer
model = CNNClassifier(input_size, output_size)


model.eval()
model.load_state_dict(torch.load('slagent.pth'))

