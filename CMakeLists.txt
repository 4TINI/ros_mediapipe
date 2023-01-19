cmake_minimum_required(VERSION 3.0.2)
project(ros_mediapipe)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  std_msgs
  sensor_msgs
  human_skeleton_msgs
)

# catkin specific configuration
catkin_package(
  CATKIN_DEPENDS
  std_msgs
  sensor_msgs
  human_skeleton_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

link_directories(
  ${catkin_LIBRARY_DIRS}
)
catkin_install_python(PROGRAMS scripts/ciccio.py
  DESTINATION /home/lfortini/mediapipe_env/lib/python3.8/site-packages
)