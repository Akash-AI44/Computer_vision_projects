import cv2

# 0 is the device index; think of it like "which camera."
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Can not open webcam")

while True:  # runs forever until a break statement stops it.
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # exit the loop when the user presses 'q'

# Always release the camera when done, or other programs can't use it.
cap.release()
cv2.destroyAllWindows()  # Closes all OpenCV windows that were opened.
