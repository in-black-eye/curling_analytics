# curling_analytics

This repository contains all the modules needed to fully analyze a curling competition, from stone detection to game visualization and creation of a final pdf document with all the necessary game statistics.

## Project Structure 
---

This repo contains the following modules:

## models
Contains weights and configs of used models.

### S3_utils
This module is needed to connect to the s3 storage with records of athletes' training and photos of the region of interest ("house")

### Keypoints_detection
This module is required for detecting key points of the field. 

### Homography
This module creates a homography matrix based on the coordinates of the key points of the field in pixels.

### Stones detection, classification and tracking 
The main module of the project. Receives frames as input and detects stones on them, determines their belonging to the team and position in pixels. 

### Visualizations 
Provides tools for visualizing various aspects of the analysis. There are: tracking of stones from bird`s eye view, changes in stones locations in the house throw-to-throw. 

### Judging_system
The core system for analyzing throws. Contains modules for evaluating throws of specific players and the team as a whole. 

### Statistics_computation 
The module combines statistics of individual players and the team as a whole into a separate file with graphs and charts for coaches. 

### Utils
A collection of common utility functions used across various modules:
