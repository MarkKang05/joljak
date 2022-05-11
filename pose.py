import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Process


cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

#global count = 0
# global toggle; #up=1 down =0
# toggle=True

# global last_state
# global current_state

toggle = True
last_state=0
current_state=0

count = 0
last_toggle_state=0
current_toggle_state=0

class Object:
    def __init__(self, size=200):
        self.imgRead = cv2.imread('./image/mm.png')
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

def calc_count(angle):


    global current_state
    global last_state
    global toggle
    global count
    if(angle<90):
        current_state = 1
    else:
        current_state = 0

    if (last_state == 0 and current_state == 1):
        count+=1
        if(toggle==0):
            toggle=1
        else:
            toggle=0
    last_state = current_state 


    # print(toggle)
    # print(count)

def counting():

    global toggle

    global last_toggle_state
    global current_toggle_state
    global count

    current_toggle_state = toggle

    if ( last_toggle_state ==0 and current_toggle_state ==1):
        count+=1
    last_toggle_state = current_toggle_state
    # print(count)
    



def cal_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)


    if angle > 180.0:
        angle = 360-angle
    return angle

# p = Process(target=counting)
# p.start()
# p.join()

# def main_def():

obj = Object()

while True:
    ret, frame = cap.read()
    #black_sc = np.zeros((640,480,3),dtype=np.uint8)
    black_sc = np.zeros((480,640,3),dtype=np.uint8)
    #flipped = cv2.flip(frame, flipCode=-1)
    #frame1 = cv2.resize(flipped, (640, 480))
    
    # flipped = cv2.flip(frame, flipCode=-1)
    frame1 = cv2.resize(frame, (640, 480))
    rgb_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_img)
    #mpDraw.draw_landmarks(black_sc, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    mpDraw.draw_landmarks(frame1, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    #cv2.imshow("Frame", rgb_black_scr)

    #cv2.imshow("Frame", frame1)
    try:
        landmarks = result.pose_landmarks.landmark
        #cal_angle()
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
        #print(shoulder, elbow, wrist)

        angle = cal_angle(shoulder, elbow, wrist)
        calc_count(angle)
        #print(angle, toggle)
        #print(landmarks)
        #print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)
    except:
        pass

    # except BaseException as err:
    #     print(f"Unexpected {err=}, {type(err)=}")
    #     raise
    
    cv2.namedWindow('black', cv2.WINDOW_NORMAL)

    obj.insert_object(frame1)
    #cv2.setWindowProperty('black', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.imshow("black", black_sc)
    cv2.putText(frame1, str(count), (30, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("black", frame1)

    #cv2.setWindowProperty('black', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# if __name__ == '__main__':
#     #info('main line')
#     g_angle = 0
#     main_p = Process(target=main_def)

#     main_p.start()

