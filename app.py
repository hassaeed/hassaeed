import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload one or more images to detect large Cocconeis (elliptical) and Lunate (half-moon) shaped diatoms.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)

def classify_shape(contour):
    if len(contour) < 5:
        return "Unknown"

    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    (center, axes, orientation) = ellipse
    major, minor = axes

    ratio = minor / major

    # Detect elliptical and half-moon (lunate) shapes based on their aspect ratio
    if 0.7 <= ratio <= 0.95:  # Cocconeis (ellipse)
        return "Cocconeis (ellipse)"
    elif 0.3 < ratio < 0.5:  # Lunate (half-moon) - Wider than tall (crescent shape)
        return "Lunate (half-moon)"
    else:
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

    # Loop through the contours to detect shapes based on the aspect ratio
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 250:  # Filter out small contours that are unlikely to be relevant
            label = classify_shape(cnt)
            if label != "Unknown":  # Only process if it's one of the shapes we care about

                # Draw the contour on the image
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

    return result

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        processed_img = process_image(img_np)

        st.subheader(f"Processed Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_container_width=True)
