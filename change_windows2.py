import cv2
import numpy as np
import keyboard

class Object:
    def __init__(self, size=200):
        self.imgRead = cv2.imread('image/mm.png')
        self.size = size
        self.img = cv2.resize(self.imgRead, (size, size))
        img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.mask = mask
        self.x = 0
        self.y = 0

    def insert_object(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.mask)] = 0
        roi += self.img


videos = "image/1.mp4"
cap = cv2.VideoCapture(0)
capture = cv2.VideoCapture(videos)

def action1():
    # cv2.destroyWindow("window")
    cv2.destroyAllWindows()
    cv2.namedWindow("window2", cv2.WINDOW_NORMAL)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame2 = capture.read()
        try:
            cv2.imshow("window2", frame2)
        except:
            break
        if cv2.waitKey(1) & 0xFF == ord('o'):
            break



while True:
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # obj.insert_object(frame)

    cv2.imshow("window", frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        action1()
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()