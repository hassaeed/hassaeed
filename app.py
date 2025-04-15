import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload one or more images to detect Cocconeis (elliptical) and Epithemia (coffee-bean) shaped diatoms.")

# File uploader to allow image uploads
uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)

def classify_shape(contour):
    """Classify shapes based on aspect ratio (bean or ellipse)."""
    if len(contour) < 5:
        return "Unknown", None
    
    # Fit ellipse for contours that can be fitted
    ellipse = cv2.fitEllipse(contour)
    (_, axes, _) = ellipse
    major, minor = axes
    ratio = minor / major  # Aspect ratio: minor/major (height/width)

    # Cocconeis (ellipse) shape: Aspect ratio near 0.75 - 0.95
    if 0.75 <= ratio <= 0.95:
        return "Cocconeis (ellipse)", (major, minor)
    
    # Epithemia (coffee-bean) shape: Aspect ratio near 0.3 - 0.6
    elif 0.3 <= ratio < 0.6:
        return "Epithemia (coffee-bean)", (major, minor)
    
    # If the shape doesn't match known types
    return "Unknown", None

def process_image(img_np):
    """Process the image to detect and classify shapes."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    counts = {"Cocconeis (ellipse)": 0, "Epithemia (coffee-bean)": 0, "Unknown": 0}
    measurements = []

    # Loop over each contour and classify the shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Ignore too small or too large contours
        if 100 < area < 5000:
            label, dimensions = classify_shape(cnt)
            counts[label] += 1

            # Calculate center for placing the label
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Display measurements on the image
                if label != "Unknown" and dimensions:
                    major, minor = dimensions
                    measurements.append({
                        'Shape': label,
                        'Area': area,
                        'Major Axis': major,
                        'Minor Axis': minor
                    })
                
                # Draw the contour and label the shape
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(result, f"{label}", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result, counts, measurements

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        processed_img, shape_counts, measurements = process_image(img_np)

        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_column_width=True)

        # Display counts for each shape
        col1, col2, col3 = st.columns(3)
        col1.metric("Cocconeis (ellipse)", shape_counts['Cocconeis (ellipse)'])
        col2.metric("Epithemia (coffee-bean)", shape_counts['Epithemia (coffee-bean)'])
        col3.metric("Unknown", shape_counts['Unknown'])

        # Display measurements in a table
        if measurements:
            st.write("### Measurements:")
            st.write("Shape | Area | Major Axis | Minor Axis")
            for measurement in measurements:
                st.write(f"{measurement['Shape']} | {measurement['Area']} | {measurement['Major Axis']} | {measurement['Minor Axis']}")
