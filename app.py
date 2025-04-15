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
        return "Unknown"

    # Fit ellipse and extract axis ratio (for elliptical shapes)
    ellipse = cv2.fitEllipse(contour)
    (_, axes, _) = ellipse
    major, minor = max(axes), min(axes)
    ratio = minor / major

    # Check for elliptical shapes based on the ratio of axes
    if 0.75 <= ratio <= 0.95:
        return "Cocconeis (ellipse)"
    # Check for coffee-bean shaped (Epithemia) based on the aspect ratio
    elif 0.3 <= ratio < 0.65:
        return "Epithemia (coffee-bean)"
    # If it does not fit the above criteria, it's an "Unknown"
    else:
        return "Unknown"

def process_image(img_np):
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to smooth out the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to detect contours (dark shapes on light background)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find the contours of the shapes in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()  # Copy of the image for displaying results
    counts = {"Cocconeis (ellipse)": 0, "Epithemia (coffee-bean)": 0, "Unknown": 0}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Consider contours with an area between 100px and 5000px
        if 100 < area < 5000:
            label = classify_shape(cnt)
            counts[label] += 1

            # Draw full shape (contour), not just the bounding box
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            # Calculate center for placing the label
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, label, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result, counts

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        processed_img, shape_counts = process_image(img_np)

        # Show results in the Streamlit interface
        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_column_width=True)

        # Display metrics of detected shapes
        col1, col2, col3 = st.columns(3)
        col1.metric("Cocconeis", shape_counts['Cocconeis (ellipse)'])
        col2.metric("Epithemia", shape_counts['Epithemia (coffee-bean)'])
        col3.metric("Unknown", shape_counts['Unknown'])
