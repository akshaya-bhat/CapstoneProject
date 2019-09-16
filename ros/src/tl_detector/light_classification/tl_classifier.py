# default packages
import os
import random
import time
# installed packages
import rospy
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
# custom packages
from styx_msgs.msg import TrafficLight

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

class TLClassifier(object):
    def __init__(self, is_site, models_base_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')):
        """Initialize Traffic Light Classifier

        Args:
            is_site (bool): Is it running on real world data/ not on simulator

        """

        if is_site:
            graph_file = os.path.join(models_base_path, 'frozen_inference_graph_real.pb')
            self.graph = load_graph(graph_file)
        elif not is_site:
            graph_file = os.path.join(models_base_path, 'frozen_inference_graph_sim.pb')
            self.graph = load_graph(graph_file)
        # Category index for tf graph model prediction
        self.category_index = {
            1: {'id': 1, 'name': 'Green'},
            2: {'id': 2, 'name': 'Red'},
            3: {'id': 3, 'name': 'Yellow'},
            4: {'id': 4, 'name': 'off'}
            }

        self.sess = tf.Session(graph=self.graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image, score_threshold):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Flatten
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection
        (boxes, scores, classes, num) = self.sess.run(
        [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
        feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        class_name = None
        # If score is more than threshold
        if (scores > score_threshold).any():
            max_idx = scores.argmax()
            class_name = self.category_index[classes[max_idx]]['name']

        # Determine traffic light based on class name predicted
        if class_name == 'GREEN' or class_name == 'Green' or class_name == 'green':
            return TrafficLight.GREEN
        elif class_name == 'RED' or class_name == 'Red' or class_name == 'red':
            return TrafficLight.RED
        elif class_name == 'YELLOW' or class_name == 'Yellow' or class_name == 'yellow':
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN
