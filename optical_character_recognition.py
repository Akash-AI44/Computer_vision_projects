import cv2
import importlib


try:
    easyocr = importlib.import_module("easyocr")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "EasyOCR is not installed. Install it with: python -m pip install easyocr"
    ) from exc

# Reader(['en']) loads the English text-recognition model.
# You can add more language codes, e.g. ['en', 'bn'] for English + Bangla.
reader = easyocr.Reader(['en'])

img = cv2.imread("sign.jpg")
if img is None:
    raise FileNotFoundError("Could not read image: sign.jpg")

# readtext() returns a list of (bounding_box, text, confidence) tuples.
results = reader.readtext(img)

for (bbox, text, confidence) in results:
    # bbox is 4 corner points; we only need top-left and bottom-right for a rectangle.
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Detected: '{text}' (confidence: {confidence:.2f})")

cv2.imshow("OCR Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()