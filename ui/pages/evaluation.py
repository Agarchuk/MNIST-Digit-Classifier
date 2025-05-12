from matplotlib import pyplot as plt
from backend.model_handler import ModelHandler
from models.cnn_2_layers import TwoLayersCNN
from models.simple_cnn import SimpleCNN
from ui.utils.session_state_service import SessionStateService
import streamlit as st
import seaborn as sns
from evaluate import Evaluator

class EvaluationUI:
    def __init__(self):
        self.evaluator: Evaluator = SessionStateService().get_or_create_evaluator()

    def render(self):
        st.title("Evaluation Results")
        st.subheader("Model Performance on Test Data")

        model_handler: ModelHandler = SessionStateService().get_or_create_model_handler()
        st.sidebar.write("Choose a model to evaluate")
        model_name = st.sidebar.selectbox("Model", model_handler.load_list_of_trained_models())

        if model_name == "CNN with 1 conv layer":
            path = "models/mnist_cnn.pth"
        elif model_name == "CNN with 2 conv layers":
            path = "models/mnist_cnn_2_layers.pth"
        elif model_name == "CNN with 3 conv layers":
            path = "models/mnist_cnn_3_layers.pth"
        elif model_name == "Kaggle Model":
            path = "models/kaggle_model.pth"
        else:
            st.error("Model not found")
            return

        if st.button("Evaluate"):
            with st.spinner("Evaluating model on test data..."):
                acc, cm = self.evaluator.evaluate(path)
            
            st.write(f"Accuracy: {acc:.4f}")

            col1, col2 = st.columns(2)

            with col1:
                # Create figure for confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Confusion Matrix")
                
                # Display the plot in Streamlit
                st.pyplot(fig) 

            with col2:
                st.subheader("How to Read the Confusion Matrix")
                st.markdown("""
                The **Confusion Matrix** is a tool to evaluate the performance of a classification model. Here's what it shows:
                
                - **Rows** represent the **true labels** (actual digit classes 0-9).
                - **Columns** represent the **predicted labels** (what the model predicted).
                - Each **cell** shows the number of samples for a true label that were classified as a predicted label.
                - **Diagonal cells** (top-left to bottom-right) show correct predictions (true label = predicted label).
                - **Off-diagonal cells** show incorrect predictions, indicating where the model was 'confused'.
                
                For example, if the cell at row 3, column 5 has a value of 10, it means 10 images of the digit '3' were incorrectly classified as '5'.
                """)

if __name__ == "__page__":
    EvaluationUI().render() 
