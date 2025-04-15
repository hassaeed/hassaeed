import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload one or more images to detect Cocconeis (ellipse) and Epithemia (bean-shaped) diatoms.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)

def classify_shape(contour):
    if len(contour) < 5:
        return "Unknown"

    # Fit ellipse and get axis ratio
    ellipse = cv2.fitEllipse(contour)
    (_, axes, _) = ellipse
    major, minor = max(axes), min(axes)
    ratio = minor / major

    if 0.75 <= ratio <= 0.95:
        return "Cocconeis (ellipse)"
    elif 0.3 <= ratio < 0.65:
        return "Epithemia (bean shape)"
    else:
        return "Unknown"

def process_image(img_np):
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    # Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find outer contours only
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    counts = {"Cocconeis (ellipse)": 0, "Epithemia (bean shape)": 0, "Unknown": 0}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 10000:  # Filter out noise and large blobs
            label = classify_shape(cnt)
            counts[label] += 1

            # Draw outer shape
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            # Label the center
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, label, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result, counts

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)
        except Exception as e:
            st.error(f"Failed to open image {uploaded_file.name}: {e}")
            continue

        processed_img, shape_counts = process_image(img_np)

        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Cocconeis", shape_counts['Cocconeis (ellipse)'])
        col2.metric("Epithemia", shape_counts['Epithemia (bean shape)'])
        col3.metric("Unknown", shape_counts['Unknown'])
