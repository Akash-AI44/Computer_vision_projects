import cv2
import importlib


try:
    YOLO = importlib.import_module("ultralytics").YOLO
except ModuleNotFoundError as error:
    if error.name == "ultralytics":
        raise SystemExit(
            "Install the missing dependency with: python -m pip install ultralytics"
        ) from error
    raise

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # model.track() is like model() but ALSO matches objects across frames,
    # assigning a consistent ID to each one. persist=True tells it to remember
    # tracked objects between calls (i.e., across the loop iterations).
    results = model.track(frame, persist=True, verbose=False)

    annotated_frame = results[0].plot()  # IDs are drawn automatically
    cv2.imshow("Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()