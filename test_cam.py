import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print("ret:", ret)

    if not ret:
        break

    cv2.imshow("Cam Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
