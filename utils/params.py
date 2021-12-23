
__author__ = 'Majd Jamal'

import argparse

parser = argparse.ArgumentParser(description='Image Classification of fruits and vegetables')

parser.add_argument('--epochs', type = int, default=50,
	help='Epochs. Default: 50')

args = parser.parse_args()