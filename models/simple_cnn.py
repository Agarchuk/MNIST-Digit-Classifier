import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4 * 26 * 26, 10)  # Fully connected layer: input (4 channels * 26x26), output 10 classes

    def forward(self, x):
        conv1_out = self.conv1(x)  # Save output after conv1
        relu_out = self.relu(conv1_out)  # Save output after relu
        flattened = relu_out.view(relu_out.size(0), -1)  # Reshape to 1D vector
        final_out = self.fc1(flattened)  # Output for classification
        return final_out, conv1_out, relu_out
