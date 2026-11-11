"""Grayscale, Blur, and Edge Detection"""
"""almost every CV pipeline starts with some combination of \n
grayscale → blur (remove noise) → edge/feature detection. \n
This is the "hello world" of image processing."""


import cv2
img = cv2.imread("Image.jpg")

if img is None:
    raise FileNotFoundError("Could not load Image.jpg")

# cv2.cvtColor(image, code) CONVERTS color spaces.
# cv2.COLOR_BGR2GRAY tells it: "convert from BGR color to single-channel grayscale."
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.GaussianBlur(image, kernel_size, sigma) smooths/blurs the image.
# kernel_size (5,5) = the blur "brush size" — must be odd numbers. Bigger = blurrier.
# sigma=0 means "let OpenCV auto-calculate the blur intensity."
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.Canny(image, threshold1, threshold2) detects EDGES — sharp brightness changes.
# The two thresholds control sensitivity: pixels above threshold2 are definitely edges,
# below threshold1 are definitely not, and in-between depends on connectivity.
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blurred)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
