import cv2

cap = cv2.VideoCapture(0)

cv2.namedWindow("windows", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("windows", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow("windows", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()