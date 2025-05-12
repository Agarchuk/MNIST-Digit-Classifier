from backend.image_processor import ImageProcessor
from backend.model_handler import ModelHandler
from evaluate import Evaluator
from ui.utils.session_config import SessionConfig
import streamlit as st

class SessionStateService:
    """Service for managing user state in Streamlit."""

    @staticmethod
    def set(key: str, value):
        """Sets the value for the specified key in session_state."""
        st.session_state[key] = value

    @staticmethod
    def get(key: str, default=None):
        """Gets the value from session_state; returns default if not found."""
        value = st.session_state.get(key, default)
        return value
    
    @staticmethod
    def has(key: str) -> bool:
        """Checks if the key exists in session_state."""
        return key in st.session_state
    
    @staticmethod
    def get_or_create_component(key, constructor, *args, **kwargs):
        if not SessionStateService.has(key):
            component_instance = constructor(*args, **kwargs) if callable(constructor) else constructor()
            SessionStateService.set(key, component_instance)
        return SessionStateService.get(key)
    
    def get_or_create_model_handler(self):
        return SessionStateService.get_or_create_component(
            SessionConfig.MODEL_HANDLER,
            ModelHandler
        )
    
    def get_or_create_image_processor(self):
        return SessionStateService.get_or_create_component(
            SessionConfig.IMAGE_PROCESSOR,
            ImageProcessor
        )
    
    def get_or_create_evaluator(self):
        return SessionStateService.get_or_create_component(
            SessionConfig.EVALUATOR,
            Evaluator
        )
