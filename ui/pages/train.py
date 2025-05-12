from backend.model_handler import ModelHandler
from models.simple_cnn import SimpleCNN
from models.cnn_2_layers import TwoLayersCNN
from models.cnn_3_layers import ThreeLayersCNN
from train import ModelTrainer
import streamlit as st

class TrainUI:
    def render(self):
        st.title("MNIST Digit Classifier")
        st.subheader("Train a digit")

        choice = st.radio("Choose a model", ["CNN with 1 conv layer", "CNN with 2 conv layers", "CNN with 3 conv layers"])

        st.sidebar.write("Trained models")
        trained_models = ModelHandler().load_list_of_trained_models()
        if trained_models:
            for model in trained_models:
                st.sidebar.markdown(f"- {model}")
        else:
            st.sidebar.write("❌ No trained models found yet. Train a model to get started!")

        if choice == "CNN with 1 conv layer":
            model = SimpleCNN()
            path = "models/mnist_cnn.pth"
        elif choice == "CNN with 2 conv layers":
            st.sidebar.write("CNN parameters")

            st.sidebar.write("Number of filters:")
            num_filters_1 = st.sidebar.number_input("Number of filters for the first layer", min_value=4, max_value=100, value=32)
            num_filters_2 = st.sidebar.number_input("Number of filters for the second layer", min_value=4, max_value=200, value=64)
            
            batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=64)
            epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=5)

            model = TwoLayersCNN(num_filters_1, num_filters_2)
            path = "models/mnist_cnn_2_layers.pth"
        elif choice == "CNN with 3 conv layers":
            st.sidebar.write("CNN parameters")

            st.sidebar.write("Number of filters:")
            num_filters_1 = st.sidebar.number_input("Number of filters for the first layer", min_value=4, max_value=100, value=32)
            num_filters_2 = st.sidebar.number_input("Number of filters for the second layer", min_value=4, max_value=200, value=64)
            num_filters_3 = st.sidebar.number_input("Number of filters for the third layer", min_value=4, max_value=300, value=128)
            
            batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=64)
            epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=5)

            model = ThreeLayersCNN(num_filters_1, num_filters_2, num_filters_3)
            path = "models/mnist_cnn_3_layers.pth"

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                ModelTrainer.train_model(model, path, batch_size, epochs)
                st.success("Model trained successfully!")
                st.rerun()  

if __name__ == "__page__":
    TrainUI().render() 
