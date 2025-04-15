import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Shape Detector", layout="wide")
st.title("🔍 Shape Detector")
st.write("Upload one or more images to detect shapes that resemble red blood cells (gray/white in color) within the size range of 150px to 250px.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def process_image(img_np):
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply color-based filtering to focus on lighter regions (gray/white)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    
    # Remove small noise with morphological operations (e.g., opening)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    shape_count = 0

    for cnt in contours:
        # Get the bounding box for the contour (x, y, w, h)
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filter based on the area of bounding box (150px to 250px in width/height range)
        if 150 < w < 250 and 150 < h < 250:
            shape_count += 1
            # Draw all contours
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            # Calculate center for placing the label
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, "Detected Shape", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result, shape_count

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        processed_img, shape_total = process_image(img_np)

        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_column_width=True)

        st.metric("Shapes Detected", shape_total)
