import streamlit as st

class Sidebar:
    @staticmethod
    def render():
        """Render the sidebar content."""
        input_method = st.sidebar.radio(
            "Choose Input Method",
            ["Draw", "Upload"]
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.markdown("""
        This app uses a Convolutional Neural Network trained on the MNIST dataset to recognize handwritten digits.

        - Draw or upload a digit (0-9)
        - The AI will predict the digit
        - See the confidence score of the prediction
        """)
        
        return input_method 