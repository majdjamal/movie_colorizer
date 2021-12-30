
__author__ = 'Majd Jamal'

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mp4_to_jpg(movie_path = 'data/movie/train/charade.mp4', training_path = 'data/movie_frames/train/X/', label_path = 'data/movie_frames/train/Y/', training = True):
    """ Generates data of a color movie. Images in greyscale are saved
    in training_path, and colorful images as labels in label_path.
    """

    vidcap = cv2.VideoCapture(movie_path)
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    print(fps)
    file_count = 0

    while success:
        success,image = vidcap.read()

        #print('read a new frame:',success)
        if count%(int(fps)) == 0 :

             if training:
                color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(training_path + 'X%d.jpg'%file_count, color_image)

             gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             cv2.imwrite(label_path + 'Y%d.jpg'%file_count, gray_image)
             file_count += 1
             print(f'successfully written {count} frame')

             if file_count >= 10000:
                break

        count += 1
