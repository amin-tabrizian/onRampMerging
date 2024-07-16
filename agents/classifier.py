import torch.nn as nn


# Define a CNN model with a fully connected layer for classification
class CNNClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, output_size)  # Adjust the input size based on your data

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x