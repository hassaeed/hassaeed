import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Diatom Shape Detector", layout="wide")
st.title("🔬 Diatom Shape Detector")
st.write("Upload image(s) to detect Cocconeis (ellipse), Epithemia (coffee-bean), and Round diatoms.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def classify_shape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Unknown"

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Optional: Draw ellipse if contour is big enough
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (_, axes, _) = ellipse
        major, minor = max(axes), min(axes)
        ratio = minor / major
    else:
        ratio = 0

    if circularity > 0.82:
        return "Round"
    elif 0.7 <= ratio <= 0.95:
        return "Cocconeis (ellipse)"
    elif 0.3 <= ratio < 0.65:
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
    counts = {
        "Round": 0,
        "Cocconeis (ellipse)": 0,
        "Epithemia (coffee-bean)": 0,
        "Unknown": 0
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 20000:
            label = classify_shape(cnt)
            counts[label] += 1

            # Draw contour
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

            # Put label at center
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

        st.subheader(f"Results for: {uploaded_file.name}")
        st.image(processed_img, caption=f"Processed Image: {uploaded_file.name}", use_column_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Round", shape_counts["Round"])
        col2.metric("Cocconeis", shape_counts["Cocconeis (ellipse)"])
        col3.metric("Epithemia", shape_counts["Epithemia (coffee-bean)"])
        col4.metric("Unknown", shape_counts["Unknown"])
