import pygame
import time
pygame.mixer.init()
#pygame.mixer.music.load('1.mp3')
#pygame.mixer.music.play()
test_sound = pygame.mixer.Sound("/home/pi/joljak/1.wav")
test_sound.set_volume(1.0)
test_sound.play()

while True:
    time.sleep(100)
    test_sound.stop()

pygame.quit()


