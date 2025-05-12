import torch.nn as nn

class TwoLayersCNN(nn.Module):
    def __init__(self, num_filters_1, num_filters_2):
        super(TwoLayersCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=3, padding=1)  # Added padding=1
        self.bn1 = nn.BatchNorm2d(num_filters_1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=3, padding=1)  # Added padding=1
        self.bn2 = nn.BatchNorm2d(num_filters_2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_filters_2 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        conv1_out = self.conv1(x)          # 28x28 -> 28x28 (with padding)
        bn1_out = self.bn1(conv1_out)
        relu_out_1 = self.relu(bn1_out)
        pool1_out = self.pool1(relu_out_1)  # 28x28 -> 14x14
        
        conv2_out = self.conv2(pool1_out)   # 14x14 -> 14x14 (with padding)
        bn2_out = self.bn2(conv2_out)
        relu_out_2 = self.relu(bn2_out)
        pool2_out = self.pool2(relu_out_2)  # 14x14 -> 7x7
        
        flattened = pool2_out.view(pool2_out.size(0), -1)
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)
        
        return fc2_out, conv1_out, relu_out_1, conv2_out, relu_out_2
