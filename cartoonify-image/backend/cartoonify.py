import cv2
import numpy as np

def cartoonify_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found or invalid image format.")

    # 1. Resize
    img = cv2.resize(img, (640, 480))

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply median blur
    gray_blur = cv2.medianBlur(gray, 5)

    # 4. Detect edges
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    # ✅ For grayscale cartoon:
    # Instead of combining with the original color image,
    # just use the edges on top of the grayscale image.

    # Smooth the grayscale a bit more
    smooth_gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply edges as mask
    cartoon_gray = cv2.bitwise_and(smooth_gray, smooth_gray, mask=edges)

    # OpenCV expects 3 channels if you want to save/view as normal
    cartoon_gray_bgr = cv2.cvtColor(cartoon_gray, cv2.COLOR_GRAY2BGR)

    return cartoon_gray_bgr
