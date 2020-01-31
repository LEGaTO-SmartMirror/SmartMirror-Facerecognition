import os
import time
import numpy as np

import tensorflow as tf
if (tf.__version__ == "1.15.1"):
    import tensorflow.compat.v1 as tf


CUDA_VISIBLE_DEVICES=0,1

BASE_DIR = os.path.dirname(__file__) + '/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'protos/face_label_map.pbtxt'


class FaceDetector:
    def __init__(self):
        # Load models
        self.detection_graph = tf.Graph()

        config = tf.ConfigProto(device_count = {'GPU': 1})
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(graph=self.detection_graph,config=config)

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(BASE_DIR + PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess.close()

    def detect(self, image):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        elapsed_time = int((time.time() - start_time) * 1000)
        #print('[FaceDetector] Inference time cost (ms): {}'.format(elapsed_time))

        # Ratio to real position
        boxes[0, :, [0, 2]] = (boxes[0, :, [0, 2]]*image.shape[0])
        boxes[0, :, [1, 3]] = (boxes[0, :, [1, 3]]*image.shape[1])
        return np.squeeze(boxes).astype(int), np.squeeze(scores)
