from styx_msgs.msg import TrafficLight
from PIL import ImageDraw, ImageColor, Image
import tensorflow as tf
import numpy as np
import time

cmap = ImageColor.colormap
COLOR_LIST = sorted([c for c in cmap.keys()])

def load_graph(graph_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
    return graph

def filter_boxes(min_scoure, boxes, scores, classes):
    idxs = []
    for i in range(len(classes)):
        if score[i] >= min_score:
            idx.append(i)
    filtered_boxes = boxes[idx, ...]
    filtered_scores = scores[idx, ...]
    filtered_classes = classes[idx, ...]
    return filtered_boxes, filtered_scores ,filtered_classes

def convert_to_image_coordinates(boxes, height, width):
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    return box_coords

def draw_boxes(image, boxes, classes, thickness = 4):
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        drawline([(left, top), (left, bot), (right, bot), (right, top), (left, top)])

class TLClassifier(object):
    def __init__(self, CONF_THRESHOLD):
        self.graph = tf.Graph()
        self.conf = CONF_THRESHOLD
        graph_path = "light_classification/ssd_inception_v2_inference_graph.pb"
        self.detection_graph = load_graph(graph_path)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph = self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        with self.detection_graph.as_default():
            image_np = np.expend_dims(np.asarray(image, dtype = np.uint8), 0)
            time_start = time.time()
            boxes, scores, classes = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor:image_np})
            time_end = time.time()
            detection_time = (time_end - time_start)
        
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        detected_color = -1
        if scores[0] > self.conf:
            if classes[0] == 2:
                detected_color = TrafficLight.RED
            elif classes[0] == 3:
                detected_color = TrafficLight.YELLOW
            elif classes[0] == 1:
                detected_color = TrafficLight.GREEN
        else:
            detected_color = TrafficLight.UNKNOWN

        return detected_color
