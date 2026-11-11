import cv2

# cv2.imread(path) reads an image file from disk into a numpy array.
img = cv2.imread("Image.jpg")

if img is None:
    raise FileNotFoundError(
        "Could not find the Image.jpg - check the path or filename")

# img.shape tells you (height, width, channels)
print("Image shape", img.shape)

# cv2.imshow(window_name, image) opens a window and displays the image in it.
cv2.imshow("My First Image", img)

cv2.waitKey(0)  # cv2.waitKey(0) pauses the program and waits for ANY keypress.

cv2.destroyAllWindows()  # Closes all OpenCV windows that were opened.
