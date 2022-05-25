import cv2
import numpy as np
import keyboard

videos = "image/1.mp4"

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

obj = Object()

cap = cv2.VideoCapture(0)

check_window2 = False

def action1():
    cv2.namedWindow("window2", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.destroyWindow("window")
    while True:
        video_cap = cv2.VideoCapture(videos)
        global check_window2
        ret, frame = video_cap.read()
        frame = cv2.flip(frame, 1)
        try:
            cv2.imshow("window2",frame)
        except:
            print("dd")
            break

        # if cv2.waitKey(1) & 0xFF == ord('o'):
        #     check_window2 = True
        #     break

while True:
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if check_window2 == True:
        cv2.destroyWindow("window2")
        check_window2 = False

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    obj.insert_object(frame)

    cv2.imshow("window", frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        action1()
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()