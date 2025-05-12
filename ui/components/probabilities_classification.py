import streamlit as st
import torch

from ui.utils.session_state_service import SessionStateService

class ProbabilitiesClassification:
    @staticmethod
    def render():
        probabilities = SessionStateService.get('probabilities')
        st.subheader("Probabilities Classification")
        
        probs_numpy = probabilities[0].detach().numpy()
        
        chart_data = {str(i): prob for i, prob in enumerate(probs_numpy)}
        
        st.bar_chart(chart_data)
