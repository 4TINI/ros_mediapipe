#!/home/lfortini/mediapipe_env/bin/python3

import mediapipe as mp
import rospy
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from human_skeleton_msgs.msg import SkeletonArray, Skeleton, BodyPart, Pixel
from sensor_msgs.msg import Image, CameraInfo
from threading import Thread
import numpy as np
import copy
from protobuf_to_dict import protobuf_to_dict

mp_face_detection = mp.solutions.face_detection
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
num_persons = 1 #for now just single person

def _make_2d_skeleton_msg(header, pose_2d):
    mFrame = SkeletonArray()
    mFrame.skeletons = [Skeleton() for _ in range(num_persons)]
    mFrame.header = header
    body_part_count = len(pose_2d)
    
    for person in range(num_persons):
        mFrame.skeletons[person].bodyParts = [BodyPart() for _ in range(body_part_count)]
        for bp in range(body_part_count):
            mFrame.skeletons[0]
            arr = mFrame.skeletons[person].bodyParts[bp]
            arr.pixel.x = pose_2d[bp].get('x')
            arr.pixel.y = pose_2d[bp].get('y')
            arr.score = pose_2d[bp].get('visibility')
    
    return mFrame

class rosMediapipe:
    def __init__(self, frame_id, pub_topic, color_topic, info_topic):
        self.pub = rospy.Publisher(pub_topic, SkeletonArray, queue_size=1)

        self.frame_id = frame_id

        self.bridge = CvBridge()

        self.image = None
        self.received_first_image = False

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.results = None

        self.detector = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, static_image_mode=True)

        self.tf_thread = Thread(target=self.tf_thread_callback)
        self.tf_thread.start()

        # Populate necessary K matrix values for 3D pose computation.
        self.cam_info = rospy.wait_for_message(info_topic, CameraInfo)
        self.fx = self.cam_info.K[0]
        self.fy = self.cam_info.K[4]
        self.cx = self.cam_info.K[2]
        self.cy = self.cam_info.K[5]
        
        self.image_subscriber = rospy.Subscriber(color_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        """ Mediapipe skeleton dictionary
        0. Nose                17. Left pinky #1 knuckle
        1. Left eye inner      18. Right pinky #1 knuckle
        2. Left eye            19. Left index #1 knuckle  
        3. Left eye outer      20. Right index #1 knuckle
        4. Right eye inner     21. Left thumb #2 knuckle
        5. Right eye           22. Right thumb #2 knuckle
        6. Right eye outer     23. Left hip
        7. Left ear            24. Right hip
        8. Right ear           25. Left knee
        9. Mouth left          26. Right knee
        10. Mouth right        27. Left ankle
        11. Left shoulder      28. Right ankle
        12. Right shoulder     29. Left heel
        13. Left elbow         30. Right heel
        14. Right elbow        31. Left foot index
        15. Left wrist         32. Right foot index
        16. Right wrist        
        """
    
    def image_callback(self, image_msg):
        
        header = copy.copy(image_msg.header)

        # Convert images to cv2 matrices
        try:
            self.image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # image_msg = self.bridge.imgmsg_to_cv2(image_msg)
            self.received_first_image = True
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
                
        self.detect(self.image, header)
    

    def detect(self, image_msg, header):
        img_height, img_width, _ = image_msg.shape
        self.img_height, self.img_width = img_height, img_width
        
        image_msg.flags.writeable = False
        image_msg = cv2.cvtColor(image_msg, cv2.COLOR_BGR2RGB)
        self.results = self.detector.process(image_msg)
        # image_msg.flags.writeable = True
        # self.image = image_msg
        
        # self.x_min_person = img_width
        # self.y_min_person = img_height
        # self.x_max_person = 0
        # self.y_max_person = 0
            
        if hasattr(self.results.pose_landmarks, 'landmark'):
            pose_keypoints = protobuf_to_dict(self.results.pose_landmarks)
            pose_world_keypoints = protobuf_to_dict(self.results.pose_world_landmarks)
            pose_kpt = pose_keypoints.get('landmark')
            pose_world_kpt = pose_world_keypoints.get('landmark')
            
            skel_msg = _make_2d_skeleton_msg(header, pose_kpt)
            self.pub.publish(skel_msg)

        else:
            pass
            # print("nothing detected")
            
    def tf_thread_callback(self):
        while(True):
            if self.received_first_image:
                if self.image.size!=0:
                    # Draw the pose annotation on the image.
                    self.image.flags.writeable = True
                    if self.results is not None:
                        self.mp_drawing.draw_landmarks(image=self.image,
                                                        landmark_list=self.results.pose_landmarks,
                                                        connections=mp_pose.POSE_CONNECTIONS,
                                                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                    # Flip the image horizontally for a selfie-view display.
                    cv2.imshow('ROS MediaPipe', self.image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                
       
        
if __name__ == '__main__':
    
    rospy.init_node('ros_mediapipe_node', anonymous=True)
    
    # ROS params  
    # pubTopic = rospy.get_param('~pub_topic')
    color_topic = rospy.get_param('~color_topic')
    info_topic = rospy.get_param('~info_topic')
    pub_topic = rospy.get_param('~pub_topic')
    frame_id = rospy.get_param('~frame_id')
    
    if not pub_topic:
        rospy.logfatal("Missing 'pub_topic' info in launch file. Please make sure that you have executed 'run.launch' file.")

    try:
        # Start ros wrapper
        rmp = rosMediapipe(frame_id, pub_topic, color_topic, info_topic)
        rospy.spin()

    except Exception as e:
        rospy.logerr(e)
        sys.exit(-1)