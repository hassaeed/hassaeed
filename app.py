import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Roundish Shape Detector", layout="wide")
st.title("🔍 Roundish Shape Detector")
st.write("Upload image(s) to detect diatoms with round-ish shape (50–100% circular).")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def is_roundish(contour, min_circularity=0.5):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return min_circularity <= circularity <= 1.2  # allow some over-rounding due to artifacts

def process_image(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    roundish_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 20000 and is_roundish(cnt):
            roundish_count += 1
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, "Roundish", (cX - 40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result, roundish_count

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        processed_img, count = process_image(img_np)

        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"{count} round-ish shape(s) detected", use_column_width=True)
        st.metric("Roundish Shapes Detected", count)
