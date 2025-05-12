import streamlit as st

from ui.utils.session_state_service import SessionStateService

class PredictionDisplay:
    @staticmethod
    def show_prediction():
        """Display prediction results."""
        prediction = SessionStateService.get('prediction')
        confidence = SessionStateService.get('confidence')
        if prediction is not None and confidence is not None:
            st.success(f"Prediction: {prediction}")
            st.progress(confidence)
            st.markdown(f"Confidence: {confidence:.2%}")
