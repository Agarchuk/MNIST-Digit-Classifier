import torchvision.transforms as transforms

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess_image(self, image):
        """Preprocess image for model input."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to tensor and normalize
        return self.transform(image).unsqueeze(0) 