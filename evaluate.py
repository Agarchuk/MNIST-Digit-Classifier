from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch
import torchvision
from torchvision.transforms import transforms

class Evaluator:
    def evaluate(self, path):
        """Evaluate model performance on test data"""
        # Define transform to convert images to tensors
        transform = transforms.ToTensor()

        # Load MNIST test dataset
        test_data = torchvision.datasets.MNIST(root='./data', train=False, 
                                             download=True, transform=transform)
        # Create data loader for batch processing
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, 
                                                shuffle=False)

        # Load trained model and set to evaluation mode
        model = torch.load(path)
        model.eval()

        # Lists to store predictions and true labels
        all_preds = []
        all_labels = []

        # Disable gradient calculation for inference
        with torch.no_grad():
            for images, labels in test_loader:
                # Get model predictions
                outputs = model(images)
                _, predicted = torch.max(outputs[0], 1)
                # Store predictions and labels
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        # Calculate and print accuracy
        acc = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {acc:.4f}")

        # Create and display confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        return acc, cm
