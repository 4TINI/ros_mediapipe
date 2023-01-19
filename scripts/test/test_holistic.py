#!/home/lfortini/mediapipe_env/bin/python3

import cv2
import mediapipe as mp
import datetime
import numpy as np
import imutils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

frames = 0
last_time = datetime.datetime.now()

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
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
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
      image,
      results.face_landmarks,
      mp_holistic.FACEMESH_TESSELATION,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_tesselation_style())
    
    mp_drawing.draw_landmarks(
      image,
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())
    
    mp_drawing.draw_landmarks(
      image,
      results.right_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())
    
    mp_drawing.draw_landmarks(
      image,
      results.pose_landmarks,
      mp_holistic.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    
    cv2.putText(image, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Holistic', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
