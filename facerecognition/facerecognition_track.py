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

from FaceRecognitionClass import FaceRecognition

if os.path.exists("/tmp/face_recognition_image") is True:
	os.remove("/tmp/face_recognition_image")
if os.path.exists("/tmp/face_recognition_captions") is True:
	os.remove("/tmp/face_recognition_captions")

#out = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/face_recognition_image sync=false wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)
out_cap = cv2.VideoWriter('appsrc ! shmsink socket-path=/tmp/face_recognition_captions sync=false wait-for-connection=false shm-size=100000000',0, 30, (1080,1920), True)

init_frame = np.zeros((1080,1920,3), np.uint8)

#out.write(init_frame)
out_cap.write(init_frame)


FPS = 5
FPS_LAST_MAIN = 5
DISTANCE_TO_FACE = 42


#print("Calling subprocess to open gst_rtsp_server")
BASE_DIR = os.path.dirname(__file__) + '/'
#p = subprocess.Popen(['python', BASE_DIR + 'gst_rtsp_server.py'])
#pp = subprocess.Popen(['python', BASE_DIR + 'webstream.py'])

fr = FaceRecognition(False)
cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_1m ! video/x-raw, format=BGR ,height=1920,width=1080,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR ,height=1920,width=1080,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR ,height=1920,width=1080,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true", cv2.CAP_GSTREAMER)

#so_controller = ImageSMController("test")
#so_controller.connectreceiver()

def to_node(type, message):
	# convert to json and print (node helper will read from stdout)
	try:
		print(json.dumps({type: message}))
	except Exception:
		pass
	# stdout has to be flushed manually to prevent delays in the node helper communication
	sys.stdout.flush()

# Setup variables
current_user = -1
next_user = -1
identities_counter = [0]
all_detect_people = [0]
last_all_detect_people = [0]
detect_people_bb = [(0,0,0,0)]
login_timestamp = time.time()
tracker_sort = Sort(100,5)


def shutdown(self, signum):
	to_node("status", 'Shutdown: Cleaning up camera...')
	#pp.kill()
	cap.release()
	#out.release()
	out_cap.release()

	os.remove("/tmp/face_recognition_image")
	os.remove("/tmp/face_recognition_captions")

	to_node("status", 'Shutdown: Done.')
	exit()

signal.signal(signal.SIGINT, shutdown)

def check_stdin():
	global FPS
	while True:
		lines = sys.stdin.readline()
		to_node("status", "Changing: " + lines)
		data = json.loads(lines)
		if 'FPS' in data:
			FPS = data['FPS']



t = Thread(target=check_stdin)
t.start()

def writeImageToBuffer(out,image):
	out.write(image);


#cv2.namedWindow("face detection", cv2.WINDOW_NORMAL)
#cv2.namedWindow("face detection captions", cv2.WINDOW_NORMAL)

time.sleep(1)

to_node("status", "Facerecognition started...")

# Main Loop
while True:

	

	start_time = time.time()
	ret, frame = cap.read()
	next_user = current_user
		
	if ret is False:
		print("No image")
		continue

	if FPS_LAST_MAIN != FPS:
		print(type(identities_counter))
		identities_counter = [ i * int(FPS) for i in identities_counter]
		to_node("status", identities_counter)
		FPS_LAST_MAIN = FPS

	# frame = so_controller.getfixedimage()
	identities, identities_bb, confidences, new_frame, caption_frame = fr.processframe(frame)
	print(identities_bb)
	print(confidences)

	dets =[] 

	if identities_bb !=[]:
		for b, conf in zip(identities_bb, confidences):
			dets.append([b[0],b[1],b[2],b[3],int(100 * float(confidences[0]))])
			


	trackers = tracker_sort.update(np.asarray(dets))


	if identities != []:
		if int(max(identities)) + 1 > len(identities_counter):
			needed_extention = (int(max(identities)) + 1 - len(identities_counter))

			identities_counter = np.concatenate((identities_counter, ( [0] * needed_extention)), axis=None)
			all_detect_people = np.concatenate((all_detect_people, ( [0] * needed_extention)), axis=None)
			last_all_detect_people = np.concatenate((last_all_detect_people, ( [0] * needed_extention)), axis=None)
			detect_people_bb.extend( [(0,0,0,0)] * needed_extention)

	for i in range(0, len(identities_counter)):
		if str(i) in identities:
			identities_counter[i] += 1
			detect_people_bb[i] = identities_bb[identities.index(str(i))]
		elif identities_counter[i] > 0:
			identities_counter[i] -= 1

		if identities_counter[i] > (5 * FPS):
			identities_counter[i] =  (5 * FPS)
	
		if identities_counter[i] > 10:
			all_detect_people[i] = 1
		else:
			all_detect_people[i] = 0

	if not np.array_equal(all_detect_people, last_all_detect_people) :

		publish_identities = []
		publish_identities_bb = []

		for i in range(0, len(all_detect_people)):
			if (all_detect_people[i] == 1) :
				publish_identities.append(i)
				publish_identities_bb.append(detect_people_bb[i])

		if (len(publish_identities) == 0):
			publish_identities.append(-1)

		last_all_detect_people = np.copy(all_detect_people)

		to_node("detected", {"recognised_identities": publish_identities,"recognised_identities_bb": publish_identities_bb })
		#try:
		#	print(json.dumps({"recognised_identities": publish_identities,"recognised_identities_bb": publish_identities_bb }))
		#except Exception:
		#	to_node("error", "dump not working" )
		#	# stdout has to be flushed manually to prevent delays in the node helper communication
		#sys.stdout.flush()

	frame_time = time.time() - start_time
	delta = (1.0 / FPS) - frame_time
	if delta > 0:
		time.sleep(delta)
		frame_time = (1.0 / FPS) 

	cv2.putText(new_frame, str((round(1.0/frame_time,1))) + " FPS", (50,50), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=3)
	cv2.putText(caption_frame, str((round(1.0/frame_time,1))) + " FPS", (50,50), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=3)
	#cv2.putText(caption_frame, str(identities_counter[max_user]) + " counter", (50,100), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=3)

	#ImageWriteThread = Thread(target=writeImageToBuffer, args= (out,(new_frame)))
	#ImageWriteThread.start()

	for tracker in trackers:
		cv2.rectangle(new_frame, (int(tracker[0]), int(tracker[1])), (int(tracker[2]), int(tracker[3])), color=(255,50,50), thickness=2)
		cv2.putText(new_frame, "TrackID: " + str(tracker[4]), (int(tracker[0]), int(tracker[1])-40), cv2.FONT_HERSHEY_DUPLEX, fontScale=1,color=(255, 50, 50), thickness=3)

	ret_caption_frame = np.copy(caption_frame)
	out_cap.write(ret_caption_frame)
	#CapWriteThread = Thread(target=writeImageToBuffer, args= (out_cap,(caption_frame)))
	#CapWriteThread.start()

	cv2.imshow("face detection", new_frame)
	cv2.imshow("face detection captions", ret_caption_frame)
	cv2.waitKey(1)

