import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload one or more images to detect Cocconeis (elliptical) and Epithemia (coffee-bean) diatoms.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)

def classify_shape(contour):
    ellipse_score = 0
    bean_score = 0

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major, minor = axes

        ratio = minor / major
        if 0.7 <= ratio <= 0.95:
            ellipse_score += 1
        elif 0.3 < ratio < 0.6:
            bean_score += 1

    if ellipse_score > bean_score:
        return "Cocconeis (ellipse)"
    elif bean_score > ellipse_score:
        return "Epithemia (coffee-bean)"
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

    # Create a mask to highlight non-selected areas (invert contours)
    mask = np.ones_like(result) * 255  # White background
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 250:  # You can change the size range here if needed
            label = classify_shape(cnt)

            # Fill the contours with black in the mask to exclude them
            cv2.drawContours(mask, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Invert the mask to highlight the background
    inverted_mask = cv2.bitwise_not(mask)

    # Combine the original image with the inverted mask to keep only the background visible
    result = cv2.bitwise_and(img_np, inverted_mask)

    return result

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        processed_img = process_image(img_np)

        st.subheader(f"Processed Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_container_width=True)
