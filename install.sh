#!/bin/bash

mp_ros_path=`pwd`
cd $HOME

python3 -m venv $HOME/mediapipe_env && source $HOME/mediapipe_env/bin/activate
pip install -r $mp_ros_path/requirements.txt
deactivate

sed -i "s|/home/lfortini|"$HOME"|g" $mp_ros_path/scripts/test/test_hands.py
sed -i "s|/home/lfortini|"$HOME"|g" $mp_ros_path/scripts/test/test_face.py
sed -i "s|/home/lfortini|"$HOME"|g" $mp_ros_path/scripts/test/test_holistic.py
sed -i "s|/home/lfortini|"$HOME"|g" $mp_ros_path/scripts/test/test_pose.py

sed -i "s|/home/lfortini|"$HOME"|g" $mp_ros_path/scripts/ros_mediapipe.py