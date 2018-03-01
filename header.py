import cv2
import numpy as np
import glob
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import datasets	
from matplotlib.colors import ListedColormap
import math

C = 12.5
gamma = 0.50625
classes_id = [14, 34, 33, 17]
ckpt = "outputs/checkpoints/c1s_9_c1n_256_c2s_6_c2n_64_c2d_0.7_c1vl_16_c1s_5_c1nf_16_c2vl_32_lr_0.0001_rs_1--TrafficSign--1516048778.566972"
templates_dir = 'templates'

train_data_file = 'train_data'
video_input = "01.avi"
video_output = 'video_output.avi'
output = 'output.txt'

normal_width = 640
normal_height = 480
_templates = [int(i) for i in range(1,11)]
blue_template_id = [2, 3, 8]
red_template_id = [1, 4, 5, 6, 7]
blue_red_template_id = [9]
white_black_template_id = [10]

# threshold
lower_red1 = np.array([0, 80, 80])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([145, 80, 80])
upper_red2 = np.array([180, 255, 255])

lower_blue = np.array([95, 80, 80])
upper_blue = np.array([110, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])

lower_white = np.array([0, 0, 190])
upper_white = np.array([180, 50, 255])
'''
lower_gray = np.array([0, 0, 0])
upper_gray = np.array([180, 50, 255])

lower_brown = np.array([10, 10, 0])
upper_brown = np.array([20, 255, 200])
'''
# HOG feature
width = height = 32

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*width*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def load_templates():
	images = []
	for img_dir in glob.glob(templates_dir + '/*.jpg'):
		img = cv2.imread(img_dir)
		img = cv2.resize(img, (width, height))
		images.append(img)

	titles = []
	with open(templates_dir + '/titles.txt') as f:
		for line in f:
			titles.append(line.replace('\n', ''))

	print(titles)
	return images, titles
	
def load_data_file(_data_file):
	print('\tProcessing...')
	hog_descriptors = []
	labels = []
	
	with open(_data_file, 'r') as f:
		# load labels
		for x in f.readline().split(' '):
			labels.append(int(x))
		# load hog descriptors
		for line in f:
			descriptor = []
			for x in line.split(' '):
				descriptor.append([x])
			hog_descriptors.append(np.array(descriptor, dtype=np.float32))

	return hog_descriptors, labels


def execute(_notification, _func, *_args):	
	'''	
        Execute another function and calculate the execution time of that
        Parameters:
	- notification: the string that printed when the function executed 
	- func: the function name
	- *arg: all parameters of the function
	'''
	print(_notification)
	
	t_start = datetime.now()
	result = _func(*_args)
	delta = datetime.now() - t_start

	print('Time: %fs' % (delta.seconds + delta.microseconds/1E6))
	print('-----')
	return result

