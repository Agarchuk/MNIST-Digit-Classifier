from PIL import Image
import streamlit as st


class UploadComponent:
    def __init__(self):
        st.markdown("### Upload a digit image")
        self.uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    def get_image(self):
        """Get the uploaded image if available."""
        if self.uploaded_file is not None:
            image = Image.open(self.uploaded_file)
            st.image(image, caption='Uploaded Image', width=280)
            return image
        return None
