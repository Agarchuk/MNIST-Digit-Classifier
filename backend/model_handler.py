import os
import torch
import streamlit as st
from utils.logger import log_info, log_error

class ModelHandler:
    @st.cache_resource
    def load_model(path):
        """Load the trained model."""
        log_info("Loading model...")
        try:
            model = torch.load(path)
            # Set model to evaluation mode
            model.eval()
            log_info(f"Model {model} loaded successfully")
            return model
        except Exception as e:
            # st.error("⚠️ Model file not found. Please train the model first.")
            log_error(f"Model file not found", e)
            return None

    @staticmethod
    def predict(model, image_tensor):
        """Make prediction on preprocessed image."""
        if model is None:
            return None, None

        with torch.no_grad():
            output, conv1_out, relu_out = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence, conv1_out, relu_out, probabilities
    
    @staticmethod   
    def predict_2_layers(model, image_tensor):
        """Make prediction on preprocessed image."""
        if model is None:
            return None, None
        
        with torch.no_grad():
            output, conv1_out, relu_out_1, conv2_out, relu_out_2 = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence, conv1_out, relu_out_1, conv2_out, relu_out_2, probabilities
    
    @staticmethod
    def predict_3_layers(model, image_tensor):
        """Make prediction on preprocessed image."""
        if model is None:
            return None, None
        
        with torch.no_grad():
            output, conv1_out, relu_out_1, conv2_out, relu_out_2, conv3_out, relu_out_3 = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence, conv1_out, relu_out_1, conv2_out, relu_out_2, conv3_out, relu_out_3, probabilities
    
    @staticmethod
    def load_list_of_trained_models():
        """Load the list of trained models."""
        log_info("Loading list of trained models...")
        try:
            if not os.path.exists('models'):
                return []
                
            model_mapping = {
                'mnist_cnn.pth': 'CNN with 1 conv layer',
                'mnist_cnn_2_layers.pth': 'CNN with 2 conv layers',
                'mnist_cnn_3_layers.pth': 'CNN with 3 conv layers',
                'kaggle_model.pth': 'Kaggle Model'
            }
            
            models_names = []
            for file in os.listdir('models'):
                if file.endswith('.pth') and file in model_mapping:
                    models_names.append(model_mapping[file])
                    
            log_info(f"Found {len(models_names)} trained models")
            return models_names
            
        except Exception as e:
            log_error("Error loading list of trained models", e)
            return []