import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("AI Road Crack Detection & Severity Assessment")

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg","png","jpeg"])

def detect_crack(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    crack_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    severity_ratio = crack_pixels / total_pixels

    if severity_ratio < 0.01:
        severity = "Low"
    elif severity_ratio < 0.03:
        severity = "Medium"
    else:
        severity = "High"

    return edges, severity

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    edges, severity = detect_crack(image)

    st.image(edges, caption="Detected Cracks", use_column_width=True)

    st.subheader(f"Severity Level : {severity}")