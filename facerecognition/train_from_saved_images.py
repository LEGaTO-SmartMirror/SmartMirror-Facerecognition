#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from FaceRecognitionClass import FaceRecognition

# Cam properties
fps = 30.
frame_width = 1920
frame_height = 1080

# Create capture
cap = cv2.VideoCapture(4)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)

# Create Window for stream display
cv2.namedWindow('video_realtime', cv2.WINDOW_NORMAL)

# Initialize face recognizer
fr = FaceRecognition(True)

if (not cap.isOpened()):
    print ("could not open webcam")
else:

    print("Warning: face representation are build fresh!!")
    print("Train with images from /facedatabase/images/*")
    print("ID and names provided by folder name")

    fr.svm = None
    fr.images = {}
    fr.people = {}
    fr.learnFromPics()

    while 1:

        ret, frame = cap.read()

        if ret:
            rot_frame = cv2.flip(np.rot90(frame,1),1)
            identies, identies_bb, confidences, new_frame, caption_frame = fr.processframe(rot_frame)
            cv2.imshow('video_realtime',new_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("no frame was received")
            break
