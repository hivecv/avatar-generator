import os
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from streamlit_image_select import image_select
from PIL import Image

st.set_page_config(layout="wide")
cols = st.columns(2)

API_URL = os.getenv("API_URL", "http://localhost:8000")

face1_file = Path(__file__).parent / "assets" / "basic_jan.png"
face2_file = Path(__file__).parent / "assets" / "botanist_jan.png"
face3_file = Path(__file__).parent / "assets" / "scientist_jan.png"

with cols[0]:
    disk_file = st.file_uploader("Choose your face")
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    img = image_select("Choose your Jan", [face1_file, face2_file, face3_file])

with cols[1]:
    upload_file = disk_file or picture
    if upload_file:
        response = requests.post(f"{API_URL}/generate/{img.stem}", files={"file": upload_file.getvalue()})
        response.raise_for_status()
        result = Image.open(BytesIO(response.content))
        st.image(result, width=400)

        if st.button("Finalize", type="tertiary"):
            response = requests.post(f"{API_URL}/finalize/{img.stem}", files={"file": response.content})
            response.raise_for_status()
            new_result = Image.open(BytesIO(response.content))
            st.image(new_result, width=400)
