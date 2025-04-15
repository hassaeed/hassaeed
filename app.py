import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("🔬 Diatom Shape Detector")
st.write("Upload an image to detect Cocconeis (coin-like, circular) shaped diatoms.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

def is_coin_shape(contour):
    # Calculate the contour's bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the aspect ratio (height/width) for circularity
    aspect_ratio = float(w) / h

    # Check if the shape is close to circular (aspect ratio near 1)
    if 0.8 <= aspect_ratio <= 1.2:  # Circular shapes have an aspect ratio near 1
        return True
    return False

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image: simple binary thresholding
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours (only external ones)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    count = 0

    # Loop through each contour and check if it's coin-like
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Only consider contours with area between 100px and 5000px
        if 100 < area < 5000:
            if is_coin_shape(cnt):
                # Draw the contour (outer boundary)
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

                # Calculate the center for the label
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(result, "Cocconeis (coin-like)", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                count += 1

    st.image(result, caption=f"Detected {count} coin-like shapes.", use_column_width=True)
