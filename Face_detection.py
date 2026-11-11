from pathlib import Path

import cv2

# FaceDetectorYN.create() loads the pretrained face-detection neural network.
# Arguments: model path, config path ("" = not needed for this model), and the
# input size it expects — we set a placeholder here and update it per-frame below,
# since our webcam frame size needs to match exactly.
model_path = Path(__file__).with_name("face_detection_yunet.onnx")
detector = cv2.FaceDetectorYN.create(
    str(model_path), "", (320, 320)
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # The model needs to know the exact size of the frame it's about to process,
    # so we update it every loop in case the webcam resolution ever changes.
    detector.setInputSize((width, height))

    # detect() returns a tuple: (return_code, faces).
    # faces is either None (no faces found) or a numpy array — one row per face.
    # Each row holds: [x, y, w, h, then 5 pairs of facial landmark points, then a confidence score].
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # Convert the first 4 values (bounding box) to plain integers for drawing.
            x, y, w, h = face[:4].astype(int)
            # the last value is the detection confidence (0 to 1)
            confidence = face[-1]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Face {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
