from header import *
from train_capsnet import *
from image_processing import *

from docopt import docopt
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import pickle
import os

from model import ModelTrafficSign
from data_handler import get_data

def capsules_network():
	print('*' * 20)
	print('Capsules Network')
	
	execute('=' * 20 + '\nTRAIN', train_capsnet)
    
def run_model():
	print('Runing Model')
	with open("signnames.csv", "r") as f:
		signnames = f.read()
		id_to_name = { int(line.split(",")[0]):line.split(",")[1] for line in signnames.split("\n")[1:] if len(line) > 0}
    # Load the model
	capsnet = ModelTrafficSign("TrafficSign", output_folder=None)
	capsnet.load(ckpt)
	print("loadinggg...")
	#
	templates, templates_title = execute('Loading templates', load_templates)
	print(templates_title)
	inp = cv2.VideoCapture(video_input)
	video_width = int(inp.get(cv2.CAP_PROP_FRAME_WIDTH))
	video_height = int(inp.get(cv2.CAP_PROP_FRAME_HEIGHT))
	video_fps = inp.get(cv2.CAP_PROP_FPS)
	
	print('Video resolution: (' + str(video_width) + ', ' + str(video_height) + ')')
	print('Video fps:', video_fps)

	#out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (video_width, video_height))
	
	print('Video is running')
	info = []
	frame_id = 0
	while inp.isOpened():
		ret, frame = inp.read()
		frame_id += 1
		if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
			break
		
        
		#frame = cv2.resize(frame, (video_width, video_height))
		
		process_image(frame, capsnet, id_to_name, templates, templates_title, frame_id, info)
		#out.write(frame)
		
	inp.release()
	#out.release()
	cv2.destroyAllWindows()
	with open(output, 'w') as f:
		f.write(str(frame_id) + '\n') #str(len(info))
		for elem in info:
			if (elem[1] != 11):
				f.write(' '.join(str(x) for x in elem))

if __name__ == '__main__':
	# extract_video_datasets('D:\workspace\TrafficSignRecognitionAndDetection\Contest\datasets\Orginal\\abc')
	# create_train_datasets('D:\workspace\TrafficSignRecognitionAndDetection\Contest\datasets\Orginal\\abc\_10')
	# capsules_network()
#    with open("signnames.csv", "r") as f:
#        signnames = f.read()
#    id_to_name = { int(line.split(",")[0]):line.split(",")[1] for line in signnames.split("\n")[1:] if len(line) > 0}
#
##    images = []
#
#    # Read all image into the folder
##    for filename in os.listdir("from_web"):
##        img = Image.open(os.path.join("from_web", filename))
##        img = img.resize((32, 32))
##        img = np.array(img) / 255
##        images.append(img)
#
#    # Load the model
#    capsnet = ModelTrafficSign("TrafficSign", output_folder=None)
#    capsnet.load(ckpt)
#    print("loadinggg...")
    
    run_model()
#'''
#	for img_dir in glob.glob('D:\workspace\\TrafficSignRecognitionAndDetection\Contest\datasets\Images\_5\*'):
#		print(img_dir.split('.')[0] + '111' + '.jpg')
#		img = cv2.imread(img_dir)
#		img = cv2.flip(img, 1)
#		cv2.imwrite(img_dir.split('.')[0] + '_flip' + '.jpg', img)
#'''
