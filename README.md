# smartmirror-facerecognition
[MagicMirror²](https://github.com/MichMich/MagicMirror) module for face recognition. This module shall be optimized with the [LEGaTO](https://legato-project.eu/) toolkit for energy efficiency.

# Requirements
- CUDA (9.1)
- TensorFlow
- scikit-learn

(only tested with ubuntu 16.04)

# Face Recognition Trial
To try out the face recognition simply start the python script **/facerecognition/start_recognition_webcam.py**

# Structure of the TensorFlow Face Recognition

## Pipeline
Image -> FaceDetection -> CroppedFace -> FaceRecognition -> Descriptor(128D) -> FaceClassifier -> Name

## FaceDetection(mobilenetSSD)
A mobilenet SSD(single shot multibox detector) based face detector with pretrained model provided.
Ref. [https://github.com/yeephycho/tensorflow-face-detection](https://github.com/yeephycho/tensorflow-face-detection)

## FaceRecognition(FaceNet)
TensorFlow implementation of the face recognizer described in the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering".
Ref. [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)

## FaceClassification(kNN, SVM)
Classify the features generated by FaceNet with kNN or SVM.


ACKNOWLEDGEMENT

This work has been supported by EU H2020 ICT project LEGaTO, contract #780681 .
