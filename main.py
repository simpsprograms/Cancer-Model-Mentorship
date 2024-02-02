import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class CancerClassifier(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(CancerClassifier, self)._init_()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model with input size, hidden size, and output size
input_size = 30  # Replace with the actual number of input features
hidden_size = 64  # Choose an appropriate size for the hidden layer
output_size = 1   # Binary classification (cancer or non-cancer)

model = CancerClassifier(input_size, hidden_size, output_size)

# Choose a loss function and an optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print the model architecture
print(model)