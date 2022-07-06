from glob import glob
import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Process
import time
import pygame
pygame.mixer.init()
  
videos = "image/exer1.mp4"
videos2 = "image/exer2.mp4"

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
video_cap2 = cv2.VideoCapture(videos2)
is_show = False

# 사진을 기수 화면에 넣어주는 객체, 함수
class Object:
    def __init__(self, size=300):
        self.imgRead = cv2.imread('image/img3/{}.png'.format(count*10))
        self.size = size
        self.img = cv2.resize(self.imgRead, (size, size))
        img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.mask = mask
        self.x = 340
        self.y = 0

    def insert_object(self, frame):
        if is_show:
            roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
            roi[np.where(self.mask)] = 0
            roi += self.img
        else:
            return

# 첫번째 예시 운동 영상을 재생해주는 함수
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

# 두번째 예시 운동 영상을 재생해주는 함수
def action2():
    cv2.destroyAllWindows()
    cv2.namedWindow("window2", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('window2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        # global check_window2
        ret, frame = video_cap2.read()
        frame = cv2.flip(frame, 1)
        try:
            cv2.imshow("window2",frame) 
        except:
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(1) & 0xFF == ord('o'):
            cv2.destroyWindow("windows2")
            break

# 운동을 선택하는 창
def selectExercise():
    playSound("select")
    global exercise_no
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    img = cv2.imread('image/select/2.png', cv2.IMREAD_COLOR)
    cv2.destroyWindow("main")
    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('2'):
            playSound("tutorial")
            action2()
            exercise2()
            finish()
            print("select exercise 2")
            break
        if key == ord('1'):
            playSound("tutorial")
            action1()
            exercise1()
            finish()
            print("select exercise 2")
            break
        #obj2.insert_object(black_sc)

    
# 각도를 입력받아 굽혔다 폈을때 count를 1 증가시켜주는 함수
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
        playSound(str(count))
        start_time = time.time()
        if(toggle==0):
            toggle=1
        else:
            toggle=0
    last_state = current_state 

# 세 점의 위치 값을 이용하여 세 점을 이었을때의 두 선분의 각도를 계산해주는 함수
def cal_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return angle

# 휴식 시간 화면 함수
def relax():
    playSound("good")
    relax_start = time.time()
    while True:
        if (time.time() - relax_start)>5:
            cv2.destroyWindow("window")
            break

        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        img = cv2.imread('image/relax.png', cv2.IMREAD_COLOR)
        cv2.imshow('window', img)
        cv2.waitKey(1)

# 모든 동작이 끝나고 난 후 띄워주는 화면
def finish():
    playSound("finish")
    relax_start = time.time()
    while True:
        if (time.time() - relax_start)>7:
            cv2.destroyWindow("window")
            break

        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        img = cv2.imread('image/finish.png', cv2.IMREAD_COLOR)
        cv2.imshow('window', img)
        cv2.waitKey(1)

# 첫 번째 운동 인식, 출력 코드
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

        # 모션캡처 결과를 검정화면에 띄울 것인지, 실제 화면 위에 뜨울 것인지
        if black_screen:
            mpDraw.draw_landmarks(black_sc, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            mpDraw.draw_landmarks(frame1, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            # 인식한 몸의 각 포인트들을 변수에 담아놓음
            landmarks = result.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # 각도, count계산
            angle = cal_angle(shoulder, elbow, wrist)
            calc_count(angle)
        except:
            pass

        cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            

        obj = Object()

        # 2초동안만 횟수 사진 띄워주기
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

        # 10회씩 2세트 진행
        if(count==10 and set_count==1):
            set_count=0
            count=0
            break
        #첫 세트 후, 휴식시간
        elif(count==10):
            time.sleep(1)
            set_count+=1
            count=0
            relax()

# 두 번째 운동 인식, 출력 코드
def exercise2():
    global count
    global cap
    global is_show
    global set_count
    while True:
        ret, frame = cap.read()
        black_sc = np.zeros((480,640,3),dtype=np.uint8)

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
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            angle = cal_angle(hip, shoulder, elbow)
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
        if(count==10 and set_count==1):
            set_count=0
            count=0
            break
        elif(count==10):
            time.sleep(1)
            set_count+=1
            count=0
            relax()

# 비동기로 .wav 음성파일을 재생시켜주는 함수
def playSound(name):
    ts = pygame.mixer.Sound("./sound/"+name+".wav")
    ts.set_volume(1.0)
    ts.play()

mt = False
# 최초 시작화면
while True:
    if mt==False:
        playSound("start")
        # ts = pygame.mixer.Sund("/home/pi/joljak/1.wav")
        # ts.set_volume(1.0)
        # ts.play()
        mt=True
    img = cv2.imread('image/start.png', cv2.IMREAD_COLOR)
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('main', img)


    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        action1()

    if key == ord('s'):
        selectExercise()
        mt=False
        # exercise1(kdlsklk

    if key == ord("q"):
        break
