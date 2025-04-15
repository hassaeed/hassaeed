import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload one or more images to detect Cocconeis (elliptical) and Epithemia (boomerang-shaped) diatoms.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)

def classify_shape(contour):
    if len(contour) < 5:
        return "Unknown"

    # Fit ellipse to the contour for Cocconeis
    ellipse = cv2.fitEllipse(contour)
    (center, axes, orientation) = ellipse
    major, minor = axes
    ratio = minor / major

    # Cocconeis (ellipse) detection based on aspect ratio
    if 0.75 <= ratio <= 0.95:  # Elliptical shapes
        return "Cocconeis (ellipse)"
    
    # Epithemia (boomerang) - identifying the asymmetrical and curved shapes
    elif 0.5 < ratio < 0.7:  # Asymmetrical, elongated but curved shapes
        return "Epithemia (boomerang)"
    
    return "Unknown"

def process_image(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for contrast improvement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 250:  # Filter based on the size of the contour
            label = classify_shape(cnt)
            if label != "Unknown":  # Only process known shapes

                # Draw the contour on the image
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)  # Green outline for detection

    return result

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        processed_img = process_image(img_np)

        st.subheader(f"Processed Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_container_width=True)
