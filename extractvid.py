import cv2
import pandas
import os
import collections
import numpy as np
from math import floor
import matplotlib.image as img
import sys

vidname = sys.argv[1]

capture = cv2.VideoCapture(vidname + '.avi')
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

os.makedirs('./video/', exist_ok=True)

for count in range(frame_count):
	ret, frame = capture.read()
	if not ret:
		raise ValueError("Failed to load frame #{} of {}.".format(count, filename))
	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	v[count, :, :] = frame
	img.imsave(os.path.join('video/' + vidname + '_' + str(count)  + '.png'), v[count, :, :])
