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

def load_image_into_numpy_array(image):
    (im_width, im_height, _) = image.shape
    return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

class TLClassifier(object):
    def __init__(self, is_site, models_base_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')):
        """Initialize Traffic Light Classifier

        Args:
            is_site (bool): Is it running on real world data/ not on simulator

        """
        #KOKSANG TODO
        self.count = 0
        if is_site:
            graph_file = os.path.join(models_base_path, 'frozen_inference_graph_real.pb')
            self.graph = load_graph(graph_file)
        elif not is_site:
            graph_file = os.path.join(models_base_path, 'frozen_inference_graph_sim.pb')
            self.graph = load_graph(graph_file)

        self.category_index = {
            1: {'id': 1, 'name': 'Green'},
            2: {'id': 2, 'name': 'Red'},
            3: {'id': 3, 'name': 'Yellow'},
            4: {'id': 4, 'name': 'off'}
            }
    
    def get_classification(self, image, score_threshold):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.graph.as_default():
            class_name = None
            with tf.Session(graph=self.graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
                
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
                
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.graph.get_tensor_by_name('num_detections:0')

                image_np = load_image_into_numpy_array(np.asarray(image, dtype="int32"))

                image_np_expanded = np.expand_dims(image_np, axis=0)

                #time0 = time.time()

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

                #time1 = time.time()

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                if (scores > score_threshold).any():
                    max_idx = scores.argmax()
                    class_name = self.category_index[classes[max_idx]]['name']
                    print('{}'.format(class_name), scores[max_idx])

                    fx =  0.97428
                    fy =  1.73205
                    perceived_width_x = (boxes[max_idx][3] - boxes[max_idx][1]) * 800
                    perceived_width_y = (boxes[max_idx][2] - boxes[max_idx][0]) * 600

                    # ymin, xmin, ymax, xmax = box
                    # depth_prime = (width_real * focal) / perceived_width
                    perceived_depth_x = ((.1 * fx) / perceived_width_x)
                    perceived_depth_y = ((.3 * fy) / perceived_width_y )

                    estimated_distance = round((perceived_depth_x + perceived_depth_y) / 2)
                    #print("Distance (metres)", estimated_distance)
                    #print("Time in milliseconds", (time1 - time0) * 1000, "\n") 

        if class_name == 'GREEN' or class_name == 'Green':
            return TrafficLight.GREEN
        elif class_name == 'RED' or class_name == 'Red':
            return TrafficLight.RED
        elif class_name == 'YELLOW' or class_name == 'Yellow':
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN
