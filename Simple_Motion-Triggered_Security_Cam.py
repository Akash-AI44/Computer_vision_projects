import cv2
try:
    from importlib import import_module

    YOLO = import_module("ultralytics").YOLO
except (ImportError, AttributeError) as exc:
    raise SystemExit(
        "Missing dependency: install it with 'python -m pip install ultralytics'."
    ) from exc
from datetime import datetime

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Grab one frame to use as the "background" reference for motion comparison.
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # cv2.absdiff computes the pixel-by-pixel absolute difference between two frames.
    # Large differences = something moved.
    diff = cv2.absdiff(prev_gray, gray)

    # threshold() converts the difference image to pure black/white:
    # any pixel brighter than 25 becomes 255 (white), else 0 (black).
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion_detected = cv2.countNonZero(thresh) > 5000  # enough white pixels = real motion

    if motion_detected:
        # Motion found -> run the (slower) YOLO model only when needed, saving CPU.
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Save a timestamped snapshot as evidence.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"motion_{timestamp}.jpg", annotated_frame)
        cv2.putText(annotated_frame, "MOTION DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Security Cam", annotated_frame)
    else:
        cv2.imshow("Security Cam", frame)

    prev_gray = gray  # this frame becomes the "previous" for the next loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()