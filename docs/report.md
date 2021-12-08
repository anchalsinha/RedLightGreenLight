# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Provide a brief overview of the project objhectives, approach, and results.

# 1. Introduction

This section should cover the following items:

* Motivation & Objective: What are you trying to do and why? (plain English without jargon)
* State of the Art & Its Limitations: How is it done today, and what are the limits of current practice?
* Novelty & Rationale: What is new in your approach and why do you think it will be successful?
* Potential Impact: If the project is successful, what difference will it make, both technically and broadly?
* Challenges: What are the challenges and risks?
* Requirements for Success: What skills and resources are necessary to perform the project?
* Metrics of Success: What are metrics by which you would check for success?

# 2. Related Work

# 3. Technical Approach

### 3.a. Game Logic
The game presents itself as a finite state machine


### 3.b. Player Detection
The YOLOv4 model was used for person detection using the COCO dataset. YOLO (You Only Look Once) is a state-of-the-art machine learning algorithm that uses deep convolutional neural networks for object detection and identification. An alternative model, called YOLOv4-tiny, is designed for faster model predictions and requires less compute resources. Although the detection accuracy drops slightly, we are able to achieve realtime speeds for person detection to be used in the player tracking.

### 3.c. Player Tracking
Once all the players in the scene are identified, we must tag and track each player to determine who wins, loses, and to maintain the overall game state. This task is called person re-identification, meaning we must re-identify players between frames and across many frames if there is any occlusion. To do this, we utilized the DeepSORT (Deep Simple Online Realtime Tracking) algorithm which assigns a deep feature descriptor to each player and uses cosine similarity to tag and track players across frames. This model was trained using the MARS (Motion Analysis and Re-identification Set) dataset, and was able to perform in realtime speeds for our platform.

For both the YOLOv4-tiny and DeepSORT models were pre-trained with the previously mentioned datasets, allowing us to load the model architecture and weights and feed inputs.


### 3.d. Motion Detection

# 4. Evaluation and Results

# 5. Discussion and Conclusions

# 6. References
