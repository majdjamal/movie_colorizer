
__author__ = 'Majd Jamal'

import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2

dory = 'movies/movie_file.mp4'

save_X = 'movie_frames/train/X/'
save_Y = 'movie_frames/train/Y/'

vidcap = cv2.VideoCapture(dory)
count = 0
success = True
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
print(fps)
file_count = 0

while success:
    success,image = vidcap.read()
    #print('read a new frame:',success)
    if count%(int(fps)) == 0 :

         color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         cv2.imwrite(save_X + 'X%d.jpg'%file_count, color_image)

         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         cv2.imwrite(save_Y + 'Y%d.jpg'%file_count, gray_image)
         file_count += 1
         print(f'successfully written {count} frame')

         if file_count >= 5421:
            break
    count+=1
