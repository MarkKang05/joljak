from glob import glob
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

exercise_no = 0

set_count = 0
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
        # self.x = int(640/2-size/2) # 320-100
        # self.y = int(480/2-size/2) # 240-100
        self.x = 340
        self.y = 0

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
    cv2.setWindowProperty('window2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
            cv2.destroyWindow("windows2")
            break

def selectExercise():
    global exercise_no
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    img = cv2.imread('image/select/select.001.jpeg', cv2.IMREAD_COLOR)
    while True:
        cv2.imshow('image', img)
        cv2.destroyWindow("main")
        if cv2.waitKey(1) & 0xFF == ord('1'):
            exercise_no=1
            action1()
            exercise1()
            print("select exercise 1")
            break
        if cv2.waitKey(1) & 0xFF == ord('2'):
            cv2.destroyWindow("windows2")
            exercise_no=2
            print("select exercise 2")
            break
        if cv2.waitKey(1) & 0xFF == ord('3'):
            cv2.destroyWindow("windows2")
            exercise_no=3
            print("select exercise 3")
            break
        if cv2.waitKey(1) & 0xFF == ord('o'):
            cv2.destroyWindow("windows2")
            break
        #obj2.insert_object(black_sc)

    


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

def relax():
    relax_start = time.time()
    while (time.time() - relax_start)<10:
        black = np.zeros((480,640,3),dtype=np.uint8)
        cv2.putText(black, "relax", (30, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("window", black);
        cv2.waitKey(1)


def exercise1():
    global count
    global cap
    global is_show
    global set_count
    # cv2.destroyAllWindows()
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
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            

        obj = Object()

        if (time.time()-start_time)>2:
            is_show = False
        else:
            is_show = True

        if black_screen:
            obj.insert_object(black_sc)
            cv2.putText(black_sc, str(set_count+1), (30, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("window", black_sc)
        else:
            obj.insert_object(frame1)
            cv2.putText(frame1, str(set_count+1), (30, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("window", frame1)
        if cv2.waitKey(1) & 0xFF == ord('k'):
            cv2.destroyWindow("window")
            break
        if(count==10 and set_count==2):
            set_count=0
            count=0
            break
        elif(count==10):
            time.sleep(1)
            set_count+=1
            count=0
            relax()


def exercise2():
    print("hello")

    #cv2.setWindowProperty('black', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while True:
    img = cv2.imread('image/ready/ready.001.jpeg', cv2.IMREAD_COLOR)
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('main', img)


    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        action1()

    if key == ord('s'):
        selectExercise();
        # exercise1(kdlsklk

    if key == ord("q"):
        break
