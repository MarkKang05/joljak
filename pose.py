import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Process
import time
  
videos = "image/1.mp4"


mp_pose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

black_screen = True

toggle = True
last_state=0
current_state=0

count = 0
show_count=0
last_toggle_state=0
current_toggle_state=0
start_time = 0

cap = cv2.VideoCapture(0)

video_cap = cv2.VideoCapture(videos)
is_show = False

class Object:
    def __init__(self, size=300):
        self.imgRead = cv2.imread('image/img/{}.jpg'.format(count*10))
        self.size = size
        self.img = cv2.resize(self.imgRead, (size, size))
        img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.mask = mask
        self.x = int(640/2-size/2) # 320-100
        self.y = int(480/2-size/2) # 240-100

    def insert_object(self, frame):
        if is_show:
            roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
            roi[np.where(self.mask)] = 0
            roi += self.img
        else:
            return


def action1():
    cv2.destroyAllWindows()
    cv2.namedWindow("window2", cv2.WINDOW_NORMAL)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        # global check_window2
        ret, frame = video_cap.read()
        frame = cv2.flip(frame, 1)
        try:
            cv2.imshow("window2",frame) 
        except:
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(1) & 0xFF == ord('o'):
            break

def calc_count(angle):
    global current_state
    global last_state
    global toggle
    global count
    global start_time
    if(angle<90):
        current_state = 1
    else:
        current_state = 0

    if (last_state == 0 and current_state == 1):
        count+=1
        start_time = time.time()
        if(toggle==0):
            toggle=1
        else:
            toggle=0
    last_state = current_state 


def cal_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return angle

while True:
    ret, frame = cap.read()
    black_sc = np.zeros((480,640,3),dtype=np.uint8)

    #black_sc = np.zeros((640,480,3),dtype=np.uint8)
    #flipped = cv2.flip(frame, flipCode=-1)
    #frame1 = cv2.resize(flipped, (640, 480))
    # flipped = cv2.flip(frame, flipCode=-1)

    frame1 = cv2.resize(frame, (640, 480))
    rgb_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_img)

    if black_screen:
        mpDraw.draw_landmarks(black_sc, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        mpDraw.draw_landmarks(frame1, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    try:
        landmarks = result.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = cal_angle(shoulder, elbow, wrist)
        calc_count(angle)
    except:
        pass

    cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    if(count==11):
        count=0
    
    obj = Object()

    if (time.time()-start_time)>2:
        is_show = False
    else:
        is_show = True

    if black_screen:
        obj.insert_object(black_sc)
        cv2.putText(black_sc, str(count), (30, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("window", black_sc)
    else:
        obj.insert_object(frame1)
        cv2.putText(frame1, str(count), (30, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("window", frame1)

    #cv2.setWindowProperty('black', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        action1()

    if key == ord("q"):
        break
