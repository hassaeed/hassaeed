import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_image_for_curved_shapes(image_path):
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_np.copy()
    curved_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 20000:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) > 6:  # More vertices suggests curved shape
            curved_count += 1
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result, "Curved", (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return result, curved_count

# Example usage:
# output_img, count = process_image_for_curved_shapes("your_image_path.jpg")
# plt.imshow(output_img)
# plt.title(f"Detected Curved Shapes: {count}")
# plt.axis("off")
# plt.show()
