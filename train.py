import torch
import torchvision
from torchvision import transforms

from models.cnn_2_layers import TwoLayersCNN
from models.cnn_3_layers import ThreeLayersCNN

import torch.nn as nn
import torch.optim as optim


class ModelTrainer:

    @staticmethod
    def train_model(model, path, batch_size, epochs):
        transform = transforms.Compose([
            transforms.ToTensor(), # convert the image to a tensor
        ])

        train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # download the dataset
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) # create a data loader

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(epochs):
            total_loss = 0  # Initialize total loss for this epoch
            # Iterate over batches of data
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                if isinstance(model, TwoLayersCNN):
                    outputs, _, _, _, _ = model(images)  # Get model predictions (ignoring intermediate activations)
                elif isinstance(model, ThreeLayersCNN):
                    outputs, _, _, _, _, _, _ = model(images)  # Get model predictions (ignoring intermediate activations)
                else:
                    outputs, _, _ = model(images)  # Get model predictions (ignoring intermediate activations)
                loss = criterion(outputs, labels)  # Calculate loss between predictions and true labels
                
                # Backward pass and optimization
                optimizer.zero_grad()  # Clear accumulated gradients from previous batch
                loss.backward()  # Compute gradients of the loss w.r.t. model parameters
                optimizer.step()  # Update model parameters using the computed gradients
                
                # Track and report progress
                total_loss += loss.item()  # Accumulate batch loss
                if (i + 1) % 100 == 0:  # Print status every 100 batches
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        

        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss/len(train_loader):.4f}')
        
        # Save the trained model
        torch.save(model, path)
        print(f'Model saved to {path}')