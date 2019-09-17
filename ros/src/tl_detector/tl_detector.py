#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

COLOR_THRESHOLD_SIM = 0.3
COLOR_THRESHOLD_SITE = 0.3
TL_LOOKAHEAD_WPS = 100

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # To check if it is in real site or simulation
        is_site = self.config['is_site']
        self.color_threshold = None
        if is_site:
            self.color_threshold = COLOR_THRESHOLD_SITE
            self.category_index = {
            1: {'id': 1, 'name': 'Red'},
            2: {'id': 2, 'name': 'Yellow'},
            3: {'id': 3, 'name': 'Green'}
            }
        if not is_site:
            self.color_threshold = COLOR_THRESHOLD_SIM
            self.category_index = {
            1: {'id': 1, 'name': 'Green'},
            2: {'id': 2, 'name': 'Red'},
            3: {'id': 3, 'name': 'Yellow'},
            4: {'id': 4, 'name': 'off'}
            }

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        # Initiate classifier
        self.light_classifier = TLClassifier(is_site=is_site)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)        

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        # Pradeep Selvaraj
        closest_idx = self.waypoint_tree.query([pose.pose.position.x, pose.pose.position.y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        state, score = self.light_classifier.get_classification(cv_image, self.color_threshold)
        
        print('True: %d, Predicted: %d, Max score: %s' % (light.state, state, str(score)))
        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        closest_light = None
        light_wp = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose)

        #TODO find the closest visible traffic light (if one exists) 
        # Pradeep Selvaraj
        diff_from_tl = len(self.waypoints.waypoints)
        for i, light in enumerate(self.lights):
            stop_line = stop_line_positions[i]
            
            stop_line_pose = Pose()
            stop_line_pose.position.x = stop_line[0]
            stop_line_pose.position.y = stop_line[1]
            
            stop_line_pose_stamped = PoseStamped()
            stop_line_pose_stamped.header = self.pose.header
            stop_line_pose_stamped.pose = stop_line_pose
            
            wp_index = self.get_closest_waypoint(stop_line_pose_stamped)
            closest_sl_wp = wp_index - car_position
            if  closest_sl_wp >= 0 and closest_sl_wp < diff_from_tl:
                diff_from_tl = closest_sl_wp
                closest_light = light
                light_wp = wp_index

        if closest_light and diff_from_tl < TL_LOOKAHEAD_WPS:
            state = self.get_light_state(closest_light)
            return light_wp, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
