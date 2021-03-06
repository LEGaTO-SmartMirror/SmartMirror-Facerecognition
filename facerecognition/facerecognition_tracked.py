#!/usr/bin/python
# coding: utf8

import sys
import json
import time
import cv2
import signal
import numpy as np
import os
from threading import Thread
import subprocess
from sort import *
import random

from FaceRecognitionClass import FaceRecognition

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def to_node(type, message):
	# convert to json and print (node helper will read from stdout)
	try:
		print(json.dumps({type: message}))
	except Exception:
		print("failed to json")
		pass
	# stdout has to be flushed manually to prevent delays in the node helper communication
	sys.stdout.flush()

IMAGE_HEIGHT = 1920
IMAGE_WIDTH = 1080
IMAGE_STREAM_PATH = "/dev/shm/camera_image"

try:
	to_node("status", "starting with config: " + sys.argv[1])
	config = json.loads(sys.argv[1])
	if 'image_height' in config:
		IMAGE_HEIGHT = int(config['image_height'])
	if 'image_width' in config:
		IMAGE_WIDTH = int(config['image_width'])
	if 'image_stream_path' in config:
		IMAGE_STREAM_PATH = str(config['image_stream_path'])
		
		
except:
	to_node("status", "Not a valid config.. exiting!")
	#quit()
	

global global_FPS
global_FPS = 30.0
FPS_real = 10.0
achieved_FPS = 0.0
achieved_FPS_counter = 0.0
face_update_counter = 0

fr = FaceRecognition(False)

to_node("status", "face recognition initialised")

cap = cv2.VideoCapture("shmsrc socket-path="+ str(IMAGE_STREAM_PATH) +" ! video/x-raw, format=BGR, height=" + str(IMAGE_HEIGHT) + ", width="+ str(IMAGE_WIDTH) + ", framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", cv2.CAP_GSTREAMER)

def shutdown(self, signum):
	to_node("status", 'Shutdown: Cleaning up camera...')

	cap.release()

	to_node("status", 'Shutdown: Done.')
	exit()

signal.signal(signal.SIGINT, shutdown)

def convertToCenterWH(a,b,c,d):
	h = float(d - b)
	w = float(c - a)
	x = float((a + (w/2)) / IMAGE_WIDTH)
	y = float((b + (h/2)) / IMAGE_HEIGHT)
	return (x,y),(w/IMAGE_WIDTH,h/IMAGE_HEIGHT)

def check_stdin():
	global global_FPS
	while True:
		lines = sys.stdin.readline()
		data = json.loads(lines)
		to_node("status", "Changing: " + json.dumps(data))		
		if 'FPS' in data:
			global_FPS = data['FPS']


#rastering..
horizontal_division = 480.0 #270.0
vertical_division =  270.0 #480.0

t = Thread(target=check_stdin)
t.start()

#cv2.namedWindow("face detection", cv2.WINDOW_NORMAL)

time.sleep(1)

to_node("status", "Facerecognition started, but TensorFlow will allocate memory at the first run. Entering main loop.")

FaceDict = {}
last_detection_list = []

# Main Loop
while True:

	start_time = time.time()


	FPS = global_FPS

	if FPS == 0:
			time.sleep(1)
			to_node("FACE_DET_FPS", float("{0:.2f}".format(0.0)));
			continue


	ret, frame = cap.read()
	if ret is False:
		continue
	
	#identities, identities_bb, confidences, new_frame, caption_frame = fr.processframe(frame)
	trackers = fr.findFacesTracked(frame, IMAGE_HEIGHT, IMAGE_WIDTH)
	
	New_FaceDict = {}
	for k in FaceDict.keys():
		found = False
		
		for tracker in trackers:		
			if  FaceDict[k][4] == tracker[4]:
				found = True
		
		if found is True:
			New_FaceDict[k] = FaceDict[k]
		else:
			to_node("status","lost face with id: " + str(k))

	FaceDict = New_FaceDict

	for tracker in trackers:
		if tracker[4] in FaceDict:
			FaceDict[tracker[4]][0] = tracker[0]
			FaceDict[tracker[4]][1] = tracker[1]
			FaceDict[tracker[4]][2] = tracker[2]
			FaceDict[tracker[4]][3] = tracker[3]
		else:
			name, id_number, confidence = fr.identifyFacesInBB(frame,tracker)
			tracker.append(name)
			tracker.append(id_number)
			tracker.append(confidence)
			FaceDict[tracker[4]] = tracker
			to_node("status","new face with " + str(FaceDict[tracker[4]]))
			
	if (FPS_real/2) < face_update_counter:	
		face_update_counter = 0
		if (len(FaceDict) > 0):
			update_key = random.choice(list(FaceDict.keys()))
			name, id_number, confidence = fr.identifyFacesInBB(frame, FaceDict[update_key])

			if (FaceDict[update_key][6] != id_number):
				if (FaceDict[update_key][7]) > 0.0:
					FaceDict[update_key][7] -= 0.01
			else:
				if (FaceDict[update_key][7]) < 1.0:
					FaceDict[update_key][7] += 0.01

			if FaceDict[update_key][7] < 0.5:
				FaceDict[update_key][5] = name
				FaceDict[update_key][6] = id_number
				FaceDict[update_key][7] = confidence
	else:
		face_update_counter += 1

	detection_list = []

	for k in FaceDict.keys():

		center, w_h = convertToCenterWH(FaceDict[k][0],FaceDict[k][1],FaceDict[k][2],FaceDict[k][3])
		center =  int(center[0] * horizontal_division) / horizontal_division , int(center[1] * vertical_division) / vertical_division
		w_h =  int(w_h[0] * horizontal_division) / horizontal_division , int(w_h[1] * vertical_division) / vertical_division

		detection_list.append({"confidence": float("{0:.2f}".format(FaceDict[k][7])),"TrackID": int(FaceDict[k][4]) , "name": str(FaceDict[k][5]), "id": int(FaceDict[k][6]), "w_h": (float("{0:.5f}".format(w_h[0])),float("{0:.5f}".format(w_h[1]))) ,"center": (float("{0:.5f}".format(center[0])),float("{0:.5f}".format(center[1])))} )

		#cv2.rectangle(frame, (FaceDict[k][0], FaceDict[k][1]), (FaceDict[k][2], FaceDict[k][3]), color=(255,50,50), thickness=2)
		#cv2.putText(frame, "TrackID: " + str(FaceDict[k][4]), (FaceDict[k][0], FaceDict[k][1] - 40), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=2)
		#cv2.putText(frame, "Name: " + str(FaceDict[k][5]), (FaceDict[k][0], FaceDict[k][1] - 70), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=2)
		#cv2.putText(frame, "ID: " + str(FaceDict[k][6]), (FaceDict[k][0], FaceDict[k][1] - 100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=2)


	if not(not last_detection_list and not detection_list):		
	
		equality_counter = 0
		for prev_element in last_detection_list:
			for next_element in detection_list:
				if next_element["center"][0] == prev_element["center"][0] and next_element["center"][1] == prev_element["center"][1] and next_element["name"] == prev_element["name"]:
					equality_counter += 1
		
		if  not (equality_counter == len(last_detection_list) == len(detection_list)):
			to_node("DETECTED_FACES",detection_list)
			last_detection_list = detection_list

	achieved_FPS_counter += 1.0

	delta = time.time() - start_time
		
	if (1.0 / FPS) - delta > 0:
		time.sleep((1.0 / FPS) - delta)
		fps_cap = FPS
		achieved_FPS += (1.0 / FPS)
	else:
		fps_cap = 1. / delta
		achieved_FPS +=  delta

	if achieved_FPS_counter > FPS:
		FPS_real = 1 / (achieved_FPS / achieved_FPS_counter)
		to_node("FACE_DET_FPS", float("{0:.2f}".format(FPS_real)))
		achieved_FPS_counter = 0.0
		achieved_FPS = 0.0

	#cv2.putText(frame, str((round(1.0/delta,1))) + " FPS", (50,50), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=3)
	#cv2.imshow("face detection", frame)
	#cv2.waitKey(1)

