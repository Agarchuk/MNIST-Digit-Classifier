from backend.image_processor import ImageProcessor
from backend.model_handler import ModelHandler
from ui.components.drawing_digit import DrawingDigitComponent
from ui.components.feature_map import FeatureMap
from ui.components.predict import PredictionDisplay
from ui.components.probabilities_classification import ProbabilitiesClassification
from ui.components.sidebar import Sidebar
from ui.components.upload_image import UploadComponent
from ui.utils.session_state_service import SessionStateService
import streamlit as st
import torch
from utils.logger import log_info

class PredictionUI:
    def render(self):
        st.title("MNIST Digit Classifier")
        st.subheader("Predict a digit")
        st.write("This application allows you to predict handwritten digits using a trained neural network model. You can either draw a digit or upload an image to get a prediction.")
        
        model_handler: ModelHandler = SessionStateService().get_or_create_model_handler()
        image_processor: ImageProcessor = SessionStateService().get_or_create_image_processor()

        st.sidebar.markdown("### Available Models")
        st.sidebar.markdown("---")
        for model in model_handler.load_list_of_trained_models():
            st.sidebar.markdown(f"- {model}")

        model_name = st.sidebar.selectbox("Select a model", model_handler.load_list_of_trained_models())

        if model_name == "CNN with 1 conv layer":
            model = torch.load("models/mnist_cnn.pth")
        elif model_name == "CNN with 2 conv layers":    
            model = torch.load("models/mnist_cnn_2_layers.pth")
        elif model_name == "CNN with 3 conv layers":
            model = torch.load("models/mnist_cnn_3_layers.pth")
        elif model_name == "Kaggle Model":
            model = torch.load("models/kaggle_model.pth")
        else:
            st.error("Model not found")
            return     
        
        model_handler.model = model
        
        input_method = Sidebar.render()
        
        if input_method == "Draw":
            input_component = DrawingDigitComponent()
        else:
            input_component = UploadComponent()
        
        image = input_component.get_image()
            
        if image is not None and st.button('Predict'):
            SessionStateService.set('model', model)
            if input_method == "Draw":
                image = image.convert('L')
                SessionStateService.set('image', image)

                
            tensor = image_processor.preprocess_image(image)
            if model_name == "CNN with 1 conv layer":
                prediction, confidence, conv1_out, relu_out_1, probabilities = ModelHandler.predict(model, tensor)
                SessionStateService.set('prediction', prediction)
                SessionStateService.set('confidence', confidence)
                SessionStateService.set('conv1_out', conv1_out)
                SessionStateService.set('relu_out_1', relu_out_1)
                SessionStateService.set('probabilities', probabilities)
            elif model_name == "CNN with 2 conv layers":
                prediction, confidence, conv1_out, relu_out_1, conv2_out, relu_out_2, probabilities = ModelHandler.predict_2_layers(model, tensor)
                SessionStateService.set('prediction', prediction)
                SessionStateService.set('confidence', confidence)
                SessionStateService.set('conv1_out', conv1_out)
                SessionStateService.set('relu_out_1', relu_out_1)
                SessionStateService.set('conv2_out', conv2_out)
                SessionStateService.set('relu_out_2', relu_out_2)
                SessionStateService.set('probabilities', probabilities)
            elif model_name == "CNN with 3 conv layers" or model_name == "Kaggle Model":
                prediction, confidence, conv1_out, relu_out_1, conv2_out, relu_out_2, conv3_out, relu_out_3, probabilities = ModelHandler.predict_3_layers(model, tensor)
                SessionStateService.set('prediction', prediction)
                SessionStateService.set('confidence', confidence)
                SessionStateService.set('conv1_out', conv1_out)
                SessionStateService.set('relu_out_1', relu_out_1)
                SessionStateService.set('conv2_out', conv2_out)
                SessionStateService.set('relu_out_2', relu_out_2)
                SessionStateService.set('conv3_out', conv3_out)
                SessionStateService.set('relu_out_3', relu_out_3)
                SessionStateService.set('probabilities', probabilities)
           

        if SessionStateService.get('prediction') is not None:
            PredictionDisplay.show_prediction()
            FeatureMap().render()
            ProbabilitiesClassification().render()

if __name__ == "__page__":
    PredictionUI().render() 
