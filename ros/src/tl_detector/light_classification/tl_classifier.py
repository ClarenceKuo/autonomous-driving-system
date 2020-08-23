
import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self,ConfidenceThreshold):
        self.detection_confidence = ConfidenceThreshold
        GraphFilePath_SSD = 'light_classification/model/inception_v2.pb'
        self.detection_graph = self.load_graph(GraphFilePath_SSD)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # detection result
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_classification(self, image):     
        with self.detection_graph.as_default():
            image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
            boxes, scores, classes = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], \
                                                        feed_dict={self.image_tensor: image_np})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        detected_colour = TrafficLight.UNKNOWN
        
        if (scores[0] > self.detection_confidence):
            if (classes[0] == 2):
                detected_colour = TrafficLight.RED
            elif (classes[0] == 3):
                detected_colour = TrafficLight.YELLOW
            elif (classes[0] == 1):
                detected_colour = TrafficLight.GREEN
        else:
            detected_colour = TrafficLight.UNKNOWN
        
        return detected_colour