import os
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_image_select import image_select
from PIL import Image

st.set_page_config(layout="wide")
cols = st.columns(2)

API_URL = os.getenv("API_URL", "http://localhost:8000")

face1_file = Path(__file__).parent / "assets" / "basic_jan.png"
face2_file = Path(__file__).parent / "assets" / "botanist_jan.png"
face3_file = Path(__file__).parent / "assets" / "scientist_jan.png"

@st.cache_data
def generate_image(url: str, source: bytes):
    response = requests.post(url=url, files={"file": source})
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def image_to_byte_array(image) -> bytes:
    if isinstance(image, UploadedFile):
        return image.getvalue()
    else:
        buffer = BytesIO()
        image.save(buffer, format=image.format)
        return buffer.getvalue()

with cols[0]:
    st.session_state['face_file'] = st.file_uploader("Choose your face")
    enable = st.checkbox("Enable camera")
    st.session_state['upload_file'] = st.camera_input("Take a picture", disabled=not enable)
    faces = [face1_file, face2_file, face3_file]
    selected = image_select("Choose your Jan", faces)
    if st.session_state.get('selected_jan') != selected:
        generate_image.clear()
        st.session_state['selected_jan'] = selected

with cols[1]:
    if (st.session_state.get('face_file') or st.session_state.get('upload_file')) and st.session_state.get('selected_jan') and st.button("Generate", type="tertiary"):
        st.session_state['generated_file'] = generate_image(
            url=f"{API_URL}/generate/{st.session_state.get('selected_jan').stem}",
            source=image_to_byte_array(st.session_state.get('face_file') or st.session_state.get('upload_file'))
        )

    if st.session_state.get('generated_file'):
        st.image(st.session_state.get('generated_file'), width=1024)

    if st.session_state.get('selected_jan') and st.session_state.get('generated_file') and st.button("Finalize Generation", type="tertiary"):
        st.session_state['finalized_file'] = generate_image(
            url=f"{API_URL}/finalize/{st.session_state.get('selected_jan').stem}",
            source=image_to_byte_array(st.session_state.get('generated_file'))
        )

    if (st.session_state.get('face_file') or st.session_state.get('upload_file')) and st.button("Use my face", type="tertiary"):
        st.session_state['finalized_file'] = generate_image(
            url=f"{API_URL}/finalize/me",
            source=image_to_byte_array(st.session_state.get('face_file') or st.session_state.get('upload_file'))
        )

    if st.session_state.get('finalized_file'):
        st.image(st.session_state.get('finalized_file'), width=1024)
