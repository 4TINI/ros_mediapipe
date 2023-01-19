#!/home/lfortini/mediapipe_env/bin/python3

import cv2
import mediapipe as mp
import datetime
import numpy as np
import imutils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

frames = 0
last_time = datetime.datetime.now()

with mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    while(True):
        success, image = cap.read()
        image = imutils.resize(image, width=400)
        if not success:
            print("Ignoring empty camera frame.")
            continue
    
        frames += 1

        # compute fps: current_time - last_time
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Flip the image horizontally for a selfie-view display.
        cv2.putText(image, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()