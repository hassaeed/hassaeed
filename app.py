import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("🔬 Diatom Shape Detector")
st.write("Upload an image to detect Cocconeis (ellipse) and Epithemia (coffee-bean) shaped diatoms.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

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

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to highlight relevant regions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Use morphological closing to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours (only external ones)
    contours, _ = cv2.findContours(morph_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    count = 0

    # Loop through each contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Only consider contours with area between 100px and 5000px
        if 100 < area < 5000:
            label = classify_shape(cnt)

            # Draw the contour (outer boundary)
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            # Calculate the center for the label
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, label, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            count += 1

    st.image(result, caption=f"Detected {count} objects.", use_column_width=True)
