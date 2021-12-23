
__author__ = 'Majd Jamal'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.colorization import Colorizer
from utils.params import args


data_color = np.load('data/processed_data/X_val.npy') / 255
data_grey = np.load('data/processed_data/y_val.npy') / 255


data_color = np.array_split(data_color, 28)
data_grey = np.array_split(data_grey, 28)

#print(X_val.shape)
#x_test = X_val[5]
#y_test = y_val[5]


clr = Colorizer(args)
clr.main(data_grey, data_color)
