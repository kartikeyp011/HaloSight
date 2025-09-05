import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
if not cap.isOpened():
    raise RuntimeError("Could not open default camera (index 0). Try index 1 or 2.")

print("Press ESC to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Camera Check", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
