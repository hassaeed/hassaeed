import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Shape Detector", layout="wide")
st.title("🔍 Shape Detector")
st.write("Upload one or more images to detect all available shapes in the image.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def process_image(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    shape_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Adjust the area thresholds as needed
        if 50 < area < 50000:  # Lower limit 50 and upper limit 50000 (you can adjust this based on image size)
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
