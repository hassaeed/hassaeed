import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload one or more images to detect Cocconeis (elliptical) and Epithemia (coffee-bean) diatoms.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def classify_shape(contour):
    if len(contour) < 5:
        return None  # Return None if the contour is too small to classify

    # Fit ellipse and extract axis ratio
    ellipse = cv2.fitEllipse(contour)
    (_, axes, _) = ellipse
    major, minor = axes
    ratio = minor / major

    # Cocconeis detection: elliptical shapes with a ratio between 0.75 to 0.95
    if 0.75 <= ratio <= 0.95:
        return "Cocconeis (ellipse)"
    
    # Epithemia detection: coffee-bean shapes with a ratio between 0.3 to 0.65
    elif 0.3 <= ratio < 0.65:
        return "Epithemia (coffee-bean)"
    
    return None  # If shape doesn't match either Cocconeis or Epithemia

def process_image(img_np):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Improve contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply Gaussian Blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to detect edges in the image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    counts = {"Cocconeis (ellipse)": 0, "Epithemia (coffee-bean)": 0}

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter by contour area to focus on relevant sizes
        if 100 < area < 10000:
            label = classify_shape(cnt)
            if label:  # Only process if it's either Cocconeis or Epithemia
                counts[label] += 1
                # Draw the contour with a green outline
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

    return result, counts

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        # Process the image to detect Cocconeis and Epithemia shapes
        processed_img, shape_counts = process_image(img_np)

        # Display the results
        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_container_width=True)

        # Display the count of detected Cocconeis and Epithemia
        col1, col2 = st.columns(2)
        col1.metric("Cocconeis (ellipse)", shape_counts['Cocconeis (ellipse)'])
        col2.metric("Epithemia (coffee-bean)", shape_counts['Epithemia (coffee-bean)'])
