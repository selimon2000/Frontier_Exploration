## Project Description
This project focuses on developing an autonomous robot for exploring a simulated Martian cave using ROS1 and Python. The robot is equipped with advanced perception and planning capabilities to navigate unknown environments and identify & navitage to artefacts of interest.

The rover which is being navigate is omnidirectional, with an onboard RGB, Depth Camera, and laser scanner. The laser scanner generates an Occupancy Grid, which allows for frontiers to be filtered through the DBSCAN algorithm. The Occupancy Grid is then used by the MoveBaseAction to autonavigate to valid locations.

Key features include:
- Frontier-Based Exploration: Efficiently explores the environment using a cost-optimized frontier exploration algorithm.
- Artefact Detection: Utilizes YOLOv11 object detection to identify artefacts in real-time.
- Advanced Transformations: Converts 2D detections into 3D positions using depth camera data and transform libraries.
- Visualization: Displays robot navigation & detected artefacts RViz.


## How to run
- Ensure that files are downloaded into the /src directory of your specified ROS1 catkin workspace folder.
- Then build and source your workspace
  ```
  cd <your_workspace>
  catkin_make
  source devel/setup.bash
  ```
- To run the demonstration:
  ```
  roscore
  roslaunch cave_explorer cave_explorer_main.launch
  ```

## Video Demonstration:
[![YouTube](http://i.ytimg.com/vi/iWjjSNV_2g0/hqdefault.jpg)](https://www.youtube.com/watch?v=iWjjSNV_2g0)
