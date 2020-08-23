#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_tree = None
        self.imgcount = self.red_state_cnt = self.green_state_cnt = 0
        self.fp = open("/home/student/Desktop/autonomous-driving-system/ros/detect_log.txt", "w")

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

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        # Classifier
        self.light_classifier = TLClassifier(0.4)

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.loop()
    
    def loop(self):
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            if self.pose != None and self.waypoints != None and self.camera_image is not None:
                light_wp, state = self.process_traffic_lights()

                if state == TrafficLight.GREEN:
                    self.green_state_cnt += 1
                elif state == TrafficLight.RED:
                    self.green_state_cnt = 0
                    if self.red_state_cnt < STATE_COUNT_THRESHOLD * 2:
                        self.red_state_cnt += 1
                elif self.red_state_cnt > 0:
                    self.red_state_cnt -= 1
                    if self.green_state_cnt >= STATE_COUNT_THRESHOLD:
                        self.red_state_cnt = 0

                if self.red_state_cnt >= STATE_COUNT_THRESHOLD:
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))   
                else:
                    self.last_wp = -1
                    if self.red_state_cnt == 0:
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.has_image = False
                self.camera_image = None
        rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        self.waypoints_tree = KDTree(waypoints_2d)

    def traffic_cb(self, msg):
        # self.fp.write("traffic cb msg:{}\n".format(msg))
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            x,y: position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.waypoints_tree.query([x,y], 1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            diff = len(self.waypoints.waypoints)
            for i, lt in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
                d = temp_wp_idx - car_position
                if d >= 0 and d < diff:
                    diff = d
                    light = lt
                    line_wp_idx = temp_wp_idx
        # self.fp.write("process_traffic_lights reached\n")
        if light:
            state = self.get_light_state(light)
            return line_wp_idx, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
