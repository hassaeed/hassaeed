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

    ellipse_score = 0
    bean_score = 0

    ellipse = cv2.fitEllipse(contour)
    (_, axes, _) = ellipse
    major, minor = axes

    ratio = minor / major
    if 0.7 <= ratio <= 0.95:
        return "Cocconeis (ellipse)"
    elif 0.3 <= ratio < 0.6:
        return "Epithemia (coffee-bean)"
    else:
        return "Unknown"

def process_image(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    counts = {"Cocconeis (ellipse)": 0, "Epithemia (coffee-bean)": 0, "Unknown": 0}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:
            label = classify_shape(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            counts[label] += 1

    return result, counts

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        processed_img, shape_counts = process_image(img_np)

        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_column_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Cocconeis", shape_counts['Cocconeis (ellipse)'])
        col2.metric("Epithemia", shape_counts['Epithemia (coffee-bean)'])
        col3.metric("Unknown", shape_counts['Unknown'])
