import streamlit as st

from ui.pages.prediction import PredictionUI
from ui.pages.evaluation import EvaluationUI

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="wide")

class UIRouter:
    def route(self):
        pg = st.navigation([
            st.Page("ui/pages/train.py", title="Train", icon="🎯"),
            st.Page("ui/pages/prediction.py", title="Prediction", icon="🔢"),
            st.Page("ui/pages/evaluation.py", title="Evaluation", icon="📊"),
        ])

        pg.run()

