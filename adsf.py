import time
import multiprocessing
from playsound import playsound
# playsound( "1.mp3" )

play=multiprocessing.Process(target=playsound, args=("1.mp3"))
play.start()
play.terminate()

in_sec = input("시간을 입력하세요.(초):")
sec = int(in_sec)
print(sec)


#while은 반복문으로 sec가 1이 되면 반복을 멈춤
while (sec != 1 ):
    sec = sec-1
    time.sleep(1)
    print(sec)
    play=multiprocessing.Process(target=playsound, args=("{}.mp3".format(sec)))
    play.start()
    play.terminate()