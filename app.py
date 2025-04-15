import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload image(s) to detect Cocconeis (elliptical) and Epithemia (bean-like) diatoms.")

uploaded_files = st.file_uploader(
    "Upload image(s)", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True
)

def classify_shape(contour):
    if len(contour) < 5:
        return "Unknown", None, None

    ellipse = cv2.fitEllipse(contour)
    (_, axes, _) = ellipse
    major, minor = max(axes), min(axes)
    ratio = minor / major

    if 0.7 <= ratio <= 0.95:
        return "Cocconeis (ellipse)", major, minor
    elif 0.3 <= ratio < 0.65:
        return "Epithemia (bean)", major, minor
    else:
        return "Unknown", major, minor

def process_image(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    counts = {"Cocconeis (ellipse)": 0, "Epithemia (bean)": 0, "Unknown": 0}
    measurements = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 150 < area < 20000:
            label, major, minor = classify_shape(cnt)
            counts[label] += 1

            if major is not None and minor is not None:
                measurements.append((label, round(major), round(minor)))

            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                text = f"{label}"
                if major and minor:
                    text += f"\n{int(major)}x{int(minor)} px"
                cv2.putText(result, text, (cX - 40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return result, counts, measurements

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        processed_img, shape_counts, measurements = process_image(img_np)

        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption="Processed Image", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Cocconeis (ellipse)", shape_counts['Cocconeis (ellipse)'])
        col2.metric("Epithemia (bean)", shape_counts['Epithemia (bean)'])
        col3.metric("Unknown", shape_counts['Unknown'])

        if measurements:
            st.markdown("### 📏 Measurements (in pixels)")
            for i, (label, major, minor) in enumerate(measurements, 1):
                st.write(f"{i}. {label}: {major} × {minor} px")
