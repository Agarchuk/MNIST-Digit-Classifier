import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

class DrawingDigitComponent:
    def __init__(self):
        st.markdown("### Draw a digit here")
        self.canvas_result = st_canvas(
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )


    def get_image(self):
        """Get the drawn image if available."""
        if self.canvas_result.image_data is not None:
            return Image.fromarray(self.canvas_result.image_data.astype('uint8'))
        return None