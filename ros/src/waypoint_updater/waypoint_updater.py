#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

# TODO: testing code, remove after "tl_detector" node is implemented
import yaml
from styx_msgs.msg import TrafficLightArray, TrafficLight

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

TRAFFIC_LIGHT_CHECK_DIST = 200 # Testing code
LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5 # Actually below how much the car can brake, may want to increase it if yellow lights changing to red causes problem

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.vehicle_traffic_lights_cb) # TODO: ground true for testing, should be changed to /traffic_waypoint after "tl_detector" node is implemented
        # rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_lane = None # this is named as base_waypoints in the "Waypoint Updater Partial Walkthrough"
        self.waypoints_2d = None
        self.waypoint_tree = None
        
        self.stopline_wp_idx = -1
        
        # TODO: testing code, remove after "tl_detector" node is implemented
        self.lights = None
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']        

        self.loop()
        #rospy.spin()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane and self.lights: # TODO: testing code, remove "self.lights" after "tl_detector" node is implemented
                self.process_traffic_lights_groundtrue() # TODO: testing code, remove after "tl_detector" node is implemented    
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        #query will return position and index, we only need index here, so [1]
        closest_idx = self.waypoint_tree.query([x,y], 1)[1] #returns one closest point.

        #Check if the closest waypoint is ahead or behind th vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        #Equation for hyperplane through closest coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pose_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect, pose_vect-cl_vect)

        #If the dot product is positive, the car pose if ahead of hyperplane & closest idx is behind the car
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):    
        lane = Lane()
        lane.header = self.base_lane.header
        
        #Get closest and farthest waypoint
        closest_idx = self.get_closest_waypoint_idx()
        #slice the base waypoints from closest idx to LOOKAHEAD_WPS + closest_idx
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx] 
        
#         # In "Full Waypoint Walkthrough", traffic_light_check_idx = farthest_idx
#         # 
#         # Adding the TRAFFIC_LIGHT_CHECK_DIST parameter allows the node to plan
#         # a path that account for traffic light, but only calculate and publish
#         # a part of that path, increasing the performance of the node
#         #
#         # TODO: implement distance(base_lane, wp1_world, wp2_world) so that
#         # traffic_light_check_idx won't cause out of bound bug
#         traffic_light_check_idx = closest_idx + TRAFFIC_LIGHT_CHECK_DIST
        traffic_light_check_idx = farthest_idx
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= traffic_light_check_idx):
            # Don't need to stop
            lane.waypoints = base_waypoints
        else:
            rospy.loginfo('dist: %s', self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= traffic_light_check_idx))
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            
        return lane
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0) # Shift two wp to align front of the car to the stop line
        
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        
        return temp    
    
    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    # TODO: testing code, remove after "tl_detector" node is implemented
    def vehicle_traffic_lights_cb(self, msg):
        self.lights = msg.lights
    
    # TODO: testing code, remove after "tl_detector" node is implemented
    def get_closest_waypoint_basic(self, x, y):
        closest_idx = self.waypoint_tree.query([x,y], 1)[1] #returns one closest point.
        return closest_idx
    
    # TODO: testing code, remove after "tl_detector" node is implemented
    def get_light_state(self, light):
        return light.state     
    
    # TODO: testing code, remove after "tl_detector" node is implemented
    # can be recycled to "tl_detector"
    def process_traffic_lights_groundtrue(self):        
        closest_light = None
        line_wp_idx = None
        
        if (self.pose):
            # Can be behind the actual car, but the waypoints are close enough
            # for the car to stop at the light corresponding to this waypoint, 
            # with the front of the car barely crossing the line
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            car_wp_idx = self.get_closest_waypoint_basic(x, y)
            
            # Maximum waypoint for checking traffic light (may need to be bigger
            # to account for STATE_COUNT_THRESHOLD (implemented in "tl_detector")
            diff = TRAFFIC_LIGHT_CHECK_DIST
            for i, light in enumerate(self.lights):
                line = self.stop_line_positions[i]
                
                temp_wp_idx = self.get_closest_waypoint_basic(line[0], line[1])
                d = temp_wp_idx - car_wp_idx
                # The behavior is such that if the car has cross the line, 
                # it just keeps moving instead of obeying the light
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
        
        # "return" in "process_traffic_lights" function in "tl_detector" node
        if closest_light:           
            state = self.get_light_state(closest_light)
            # skipping STATE_COUNT_THRESHOLD check that is implemented in "tl_detector"          
            if (state == TrafficLight.RED):
                self.stopline_wp_idx = line_wp_idx
            else:
                self.stopline_wp_idx = -1
        else: 
            self.stopline_wp_idx = -1
    
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
