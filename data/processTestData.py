
import numpy as np
from utils.mp4_to_jpg import mp4_to_jpg

movie_path = 'data/movie/test/kansas.mp4'
saving_path = 'data/movie_frames/test/Y/'


mp4_to_jpg(movie_path, None, saving_path)
