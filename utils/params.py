
__author__ = 'Majd Jamal'

import argparse

parser = argparse.ArgumentParser(description='Colorize old movies with Pix2Pix.')

parser.add_argument('--steps', type = int, default=10000,
	help='Training iterations. Default: 10 000')

parser.add_argument('--train',  action = 'store_true', default=False,
	help='Train the network.')

parser.add_argument('--predict',  action = 'store_true', default=False,
	help='Predict colors for a old movie frame.')

parser.add_argument('--process_data',  action = 'store_true', default=False,
	help='Process training data from mp4 to npdarray, which are ready for training.')

parser.add_argument('--process_test_data',  action = 'store_true', default=False,
	help='Process test data from mp4 to jpg, which are then used for prediction.')


parser.add_argument('--test_path', type = str,  default='data/movie_frames/test/Y/Y5.jpg',
	help='Path to the image used for color prediction.')


parser.add_argument('--eta', type = float, default=2e-4,
	help='Learning Rate. Default: 0.0002')

parser.add_argument('--beta_1', type = float, default=5e-1,
	help='Beta_1 used in ADAM optimizer. Default: 0.5')

parser.add_argument('--batch_size', type = int, default=16,
	help='Batch size')


params = parser.parse_args()
