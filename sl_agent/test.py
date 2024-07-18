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
learning_rate = 0.0001
batch_size = 256
num_epochs = 100

# Generate sample data (replace with your actual data)
# For this example, we'll generate random data
x = np.load('x.npy')
print(x.shape)
# print(np.max(x))
# x = x + np.random.uniform(-0.5*np.max(x), 0.5*np.max(x))
x = np.squeeze(x).reshape(x.shape[0], -1).astype('float32') 
y = np.load('y.npy').astype('float32')

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = CustomDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model and optimizer
model = CNNClassifier(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_list = []
# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    loss_list.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
np.save('loss.npy', loss_list)
torch.save(model.state_dict(), 'slagent.pth')

# Test the model
test_dataset = CustomDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
model.load_state_dict(torch.load('slagent.pth'))

correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs).numpy()
        outputs = np.rint(outputs).astype(int)
        
        # print('out: ', outputs[5])
        # print('target: ', targets.numpy()[5])
        
        total += targets.shape[0]
        for idx, out in enumerate(outputs):
            # print(out)
            if out == targets[idx]:
                correct += 1

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy}%')
