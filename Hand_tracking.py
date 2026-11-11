import cv2
import mediapipe as mp

# The Tasks API is organized as mp.tasks -> mp.tasks.vision -> the specific task.
# These four lines are just short aliases so the rest of the code reads cleanly.
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# BaseOptions points at the model file we downloaded.
base_options = BaseOptions(model_asset_path="hand_landmarker.task")

# HandLandmarkerOptions configures the detector:
# running_mode=IMAGE tells it to treat each call as a standalone image
#   (simplest mode — good for a beginner webcam loop like this one).
# num_hands=1 -> only track one hand for simplicity.
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)

# create_from_options() builds the actual detector object from the config above.
detector = HandLandmarker.create_from_options(options)

# The 21 hand landmark connections (which point connects to which),
# used to draw the "skeleton" lines by hand since the Tasks API,
# unlike the old Legacy Solutions, doesn't include a built-in drawing helper.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                               # palm base
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # mp.Image wraps a numpy array into the format MediaPipe's Tasks API expects,
    # tagging it with its pixel format (SRGB = standard RGB here).
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # detect() runs the model on this single image and returns a result object.
    result = detector.detect(mp_image)

    height, width, _ = frame.shape

    # result.hand_landmarks is a list of hands; each hand is a list of 21 landmarks.
    for hand_landmarks in result.hand_landmarks:
        # Landmarks come back as normalized coordinates (0.0 to 1.0),
        # so we multiply by the actual frame size to get real pixel positions.
        points = [(int(lm.x * width), int(lm.y * height))
                  for lm in hand_landmarks]

        for x, y in points:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
