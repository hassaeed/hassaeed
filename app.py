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

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:
            label = classify_shape(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            count += 1

    st.image(result, caption=f"Detected {count} objects.", use_column_width=True)
