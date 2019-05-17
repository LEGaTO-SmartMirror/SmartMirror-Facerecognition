#!/usr/bin/env python2
# -*- coding: utf-8 -*-


#import txaio
#txaio.use_twisted()

import cv2
import numpy as np
import imagehash
from PIL import Image
import glob
import os
import time

#import matplotlib as mpl
#mpl.use('Agg')

from detection.FaceDetector import FaceDetector
from recognition.FaceRecognition import FaceRecognition
from classifier.FaceClassifier import FaceClassifier


if os.name == 'nt':
      SPERATOR = '\\'
else:
    SPERATOR = '/'

BASE_DIR = os.path.dirname(__file__) + SPERATOR

PROFILE_ROUND = '.4f'
FACE_DETEC_THRESHOLD = 0.5

face_detector = FaceDetector()
face_recognition = FaceRecognition()


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:10]: {}}}".format(
            str(self.identity),
            self.rep[0:10],
        )


class FaceRecognition(object):

    def __init__(self, profile_activated):
        self.profile_activated = profile_activated
        self.model_path = BASE_DIR + SPERATOR + "facedatabase" + SPERATOR
        self.svm_model_name = "trained_classifier.pkl"
        self.image_rep_name = "faces.npy"
        self.people_rep_name = "people.npy"
        self.image_folder_path = "images" + SPERATOR
        if self.profile_activated is True:
            print("FaceRecognition: initialize face recognizer")
        self.svm = None
        self.training = False
        self.training_id = 0
        self.training_name = "Unknown"
        self.trainingInProgress = False
        self.identify_threshold = 0.8
        self.face_classfier = None

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if self.profile_activated is True:
            print("FaceRecognition: loading prepresentations of kownen persons")
        try:
            self.images = np.load(self.model_path + SPERATOR + self.image_rep_name,allow_pickle=True).item()
        except IOError:
            if self.profile_activated is True:
                print("Warning: no image representation file found")
            self.images = {}
        try:
            self.people = np.load(self.model_path + SPERATOR + self.people_rep_name,allow_pickle=True).item()
        except IOError:
            if self.profile_activated is True:
                print("Warning: no people representation file found")
            self.people = {}
            self.people["0"] = "Unknown"

        if self.profile_activated is True:
            print ("FaceRecognition: try to load saved svm for estimation")
        try:
            self.face_classfier = FaceClassifier(self.model_path + SPERATOR + self.svm_model_name)
        except IOError:
            if self.profile_activated is True:
                print ("Warning: no svm saved")
            self.face_classfier = None
        if self.profile_activated is True:
            print ("FaceRecognition: initialisation done")


    def __del__(self):
        if self.profile_activated is True:
            print ("FaceRecognition: starting destruktor")


    """
    Trains the neural net with all saved images.
    Therefore every pic is loaded, cropped and representations are created.
    This uses all neuronal nets, so there need to be available (self.loadingfromimages = True)
    """
    def learnFromPics(self):
        if self.profile_activated is True:
            print ("FaceRecognition: loading representations from saved pics")
            print ("Folder: " + self.model_path + SPERATOR +self.image_folder_path)

        self.trainingInProgress = True
        self.people["0"] = "Unknown"

        time.sleep(5)

        """
        Loop trough every subfolder and get every picture
        The folder names provide id and the person name
        """
        for folder in glob.glob(self.model_path + SPERATOR + self.image_folder_path + "*"):
            args = folder.replace(self.model_path + SPERATOR + self.image_folder_path, "")
            args = args.split(SPERATOR)
            args = args[0].split("_")

            for file in glob.glob(folder + SPERATOR +"*"):
                alignedFace = []
                image = cv2.imread(file)
                image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if image.shape == (160,160,3):
                    if self.profile_activated is True:
                        print(file + " [" + str(image.shape) + "]: image of a face")
                    alignedFace.append(image)
                else:
                    bbs, scores = face_detector.detect(image)
                    bbs = bbs[np.argwhere(scores > FACE_DETEC_THRESHOLD).reshape(-1)]
                    scores = scores[np.argwhere(scores > FACE_DETEC_THRESHOLD).reshape(-1)]

                    if self.profile_activated is True:
                        print(file + ": found " + str(len(bbs)) + " faces")

                    if len(bbs) < 1:
                        continue

                    for bb in bbs:

                        cropped_face = image[bb[0]:bb[2], bb[1]:bb[3], :]
                        alignedFace.append(cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA))

                for face in alignedFace:
                    phash = str(imagehash.phash(Image.fromarray(face)))
                    rep = face_recognition.recognize(face)
                    self.images[phash] = Face(rep, int(args[0]))
                    self.people[args[0]] = args[1]

        if self.profile_activated is True:
            print ("FaceRecognition: finished loading data from saved images")
        self.trainSVM()
        self.trainingInProgress = False


    """
    Saves self.image and self.people
    """
    def saveImageDataAndTrainedNet (self):
        if self.profile_activated is True:
            print ("FaceRecognition: saving image representations and people dict")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        np.save(self.model_path + SPERATOR + self.image_rep_name, self.images)
        np.save(self.model_path + SPERATOR + self.people_rep_name, self.people)


    """
    Get the information from self.images and convert it to two list with reps and labels.
    """
    def getData(self):
        X = []
        y = []
        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)


    """
    Called to train the neural net with all representations currently loaded in self.image.
    Saves also everything afterwards.
    """
    def trainSVM(self):
        self.trainingInProgress = True
        if self.profile_activated is True:
            print ("FaceRecognition: Training SVM on {} labeled images.".format(len(self.images)))
        d = self.getData()
        if d is None:
            self.face_classfier = None
            if self.profile_activated is True:
                print ("FaceRecognition: at least 2 persons needed ..")
                self.trainingInProgress = False
            return
        else:
            (X, y) = d
            numIdentities = len(set(y + [-1]))
            if numIdentities <= 1:
                self.trainingInProgress = False
                return

            self.face_classfier = FaceClassifier()
            self.face_classfier.train(X, y, model='svm', save_model_path=(self.model_path + SPERATOR + self.svm_model_name))
            self.saveImageDataAndTrainedNet()
            print ("training done!")
            self.trainingInProgress = False


    """
    Called to process every frame (basically the main function)
    Need a Frame to process, returns die found identities, confidences and a frame with marks
    """
    def processframe(self, frame):

        #For profiling this script
        if self.profile_activated is True:
            start_time = time.time()

        if self.trainingInProgress is True:
            return [],[], frame , caption_frame

        identities = []
	identities_bb = []
        confidences = []
        caption_frame = np.zeros(frame.shape, np.uint8)


        """
        Searching for everything that look similar to a face
        Remove scores that are too low
        """
        bbs, scores = face_detector.detect(frame)
        bbs = bbs[np.argwhere(scores > 0.25).reshape(-1)]
        scores = scores[np.argwhere(scores > 0.25).reshape(-1)]

        if self.training is True:
            if len(bbs) is not 1:
                if self.profile_activated is True:
                    print ("FaceRecognition: Need to find exacly one Person")
                return [],[], frame, caption_frame
            else:
                if self.profile_activated is True:
                    print ("FaceRecognition: I want to train.. Taking Picture! Please move to differnt angles")
                pass

        # For profiling this script
        if self.profile_activated is True:
            boundingboxes_time = time.time()
            loopstart_time = []
            alignedFace_time = []
            finished_time = []
            found_identity_time = []

        """
        Deside the identity of every boxes found in the frame
        """
        for index,bb in enumerate(bbs):

            # For profiling this script
            if self.profile_activated is True:
                loopstart_time.append(time.time())

            confidence = ""
            identity = "0"

            """
            cropp every face to a 160x160 dimention to pass it into a neuronal net
            """
            cropped_face = frame[bb[0]:bb[2], bb[1]:bb[3], :]
            alignedFace = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)

            # For profiling this script
            if self.profile_activated is True:
                alignedFace_time.append(time.time())

            if alignedFace is None:
                # For profiling this script
                if self.profile_activated is True:
                    finished_time.append(time.time())
                continue

            """
            We know the identity if the picture was taken before or is equal to a used image.
            Otherwise get a feature representation of that face and compare via a neuronal net.
            """
            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
            if phash in self.images:
                 identity = str(self.images[phash].identity)
                 confidence = "1.0000"

                 # For profiling this script
                 if self.profile_activated is True:
                     found_identity_time.append(time.time())

            else:
                """if we are training a the face_recognition nn can be occupied!"""
                if self.trainingInProgress is True:
                    return [],[], frame , caption_frame, 
                rep = face_recognition.recognize(alignedFace)

                if self.training is True:

                    """
                    If training is enabled, add the representation to self.images
                    and save the new image into a subfolder of the image folder for later training runs.
                    The new image is shown in the top left.
                    """
                    self.images[phash] = Face(rep, self.training_id)
                    self.people[str(self.training_id)] = self.training_name

                    person_path = self.model_path + SPERATOR + self.image_folder_path + SPERATOR + str(self.training_id) + "_" + str(self.training_name) + SPERATOR
                    if not os.path.exists(person_path):
                        os.makedirs(person_path)

                    cv2.imwrite( person_path + str(phash) +".png",alignedFace)

                    frame[0:160, 0:160, 0] = alignedFace[:, :, 2]
                    frame[0:160, 0:160, 1] = alignedFace[:, :, 1]
                    frame[0:160, 0:160, 2] = alignedFace[:, :, 0]
                    cv2.rectangle(frame, (161, 161), (0, 0), color=(255, 50, 50), thickness=4)

                    # For profiling this script
                    if self.profile_activated is True:
                        found_identity_time.append(time.time())

                elif self.face_classfier is not None:
                    """
                    If a classifier is existent, get a prediction who this face might be.
                    Compare the confidence to a threshold to evade to low scores.
                    """
                    rep = rep.reshape(1, -1)
                    predictions = self.face_classfier.classify(rep).ravel()
                    maxI = np.argmax(predictions)
                    confidence = str(predictions[maxI].round(5))
                    if predictions[maxI] <= self.identify_threshold:
                        maxI = 0
                    identity = str(maxI)

                    # For profiling this script
                    if self.profile_activated is True:
                        found_identity_time.append(time.time())
                else:
                    # For profiling this script
                    if self.profile_activated is True:
                        found_identity_time.append(time.time())


            """ Append the identity in any case with the highest confidence."""
            identities.append(identity)
            identities_bb.append((bb[1], bb[0], bb[3], bb[2]))
            confidences.append(confidence)

            """ Mark everyone in the frame """
            cv2.rectangle(frame, (bb[1], bb[0]), (bb[3], bb[2]), color=(255,50,50), thickness=2)
            cv2.rectangle(caption_frame, (bb[1], bb[0]), (bb[3], bb[2]), color=(255,50,50), thickness=2)

            cv2.putText(frame, self.people[identity] , (bb[1], bb[0] - 10),
                        cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(255, 50, 50), thickness=2)
            cv2.putText(frame, confidence, (bb[1], bb[2] + 30),
                        cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(255, 50, 50), thickness=2)
            cv2.putText(caption_frame, self.people[identity] , (bb[1], bb[0] - 10),
                        cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(255, 50, 50), thickness=2)
            cv2.putText(caption_frame, confidence, (bb[1], bb[2] + 30),
                        cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=(255, 50, 50), thickness=2)


            # For profiling this script
            if self.profile_activated is True:
                finished_time.append(time.time())

        # For profiling this script
        if self.profile_activated is True:
            end_time = time.time()
            profile_string = "TIMES: bounding boxes: " + str( format(boundingboxes_time - start_time,PROFILE_ROUND ))
            profile_string += ", face(aligne, recognize, annotate) {"
            for i in range(0, len(loopstart_time)):
                profile_string += " (" + str(format(alignedFace_time[i] - loopstart_time[i],PROFILE_ROUND))
                profile_string += "," + str(format(found_identity_time[i] - alignedFace_time[i],PROFILE_ROUND))
                profile_string += "," + str(format(finished_time[i] - found_identity_time[i],PROFILE_ROUND)) + ")"
            profile_string += " }"
            profile_string += ", entire: " + str(format(end_time - start_time,PROFILE_ROUND))
            profile_string += ", fps: " + str(format(1.0 / (end_time - start_time),PROFILE_ROUND))
            print (profile_string)

        return identities,identities_bb,confidences,frame, caption_frame
