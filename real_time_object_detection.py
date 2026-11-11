import cv2
try:
    from ultralytics import YOLO  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "The 'ultralytics' package is required. Install it with: "
        "python -m pip install ultralytics"
    ) from exc

# YOLO("yolov8n.pt") loads a pretrained model. "n" = nano, the smallest/fastest version
# — great for learning and for running on a normal laptop without a GPU.
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # model(frame) runs the neural network on this frame.
    # verbose=False silences the per-frame console logging.
    results = model(frame, verbose=False)

    # results is a list (one entry per image passed in — here just 1 frame).
    # results[0].plot() returns a NEW image with boxes/labels/confidence already drawn on it.
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()