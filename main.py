
__author__ = 'Majd Jamal'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.pix2pix import Pix2Pix
from tensorflow.data import Dataset
import tensorflow as tf
from utils.params import params
from data.getData import getData

if params.process_test_data:
    """ Generate test dataset.
    """
    from utils.mp4_to_jpg import mp4_to_jpg

    movie_path = 'data/movie/test/kansas.mp4'
    saving_path = 'data/movie_frames/test/Y/'

    mp4_to_jpg(movie_path, None, saving_path, False)

if params.train:
    """ Train the model.
    """
    train_dataset = getData()
    train_dataset = train_dataset.batch(params.batch_size)
else:
    train_dataset = None

colorizer = Pix2Pix(params, train_dataset)
colorizer.main()
