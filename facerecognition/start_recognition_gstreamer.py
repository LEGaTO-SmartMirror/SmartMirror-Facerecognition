#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
from FaceRecognitionClass import FaceRecognition

cv2.namedWindow('video_realtime', cv2.WINDOW_NORMAL)
cv2.namedWindow('video_realtime_cap', cv2.WINDOW_NORMAL)

fr = FaceRecognition(False)

#cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_image ! video/x-raw, format=BGR ,width=1920,height=1080,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink")
cap = cv2.VideoCapture("shmsrc socket-path=/tmp/camera_1m ! video/x-raw, format=BGR ,height=1920,width=1080,framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink")

while 1:

	ret, frame = cap.read()
	if ret:


		identies, confidences, new_frame, cap_image  = fr.processframe(frame)
		cv2.imshow('video_realtime',new_frame)	
		cv2.imshow('video_realtime_cap',cap_image)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		print("[FaceRecognition] Cannot process frame.")
		break

