# Project Proposal

## 1. Motivation & Objective

The core of this game is the accurate and precise tracking of players. In real time, our system will identify and record movements of the players in coordination with the game state (red light, green light) as it does in the show. However, we also place an emphasis on integration and deployment of our hardware and software. The peripherals consisting of cameras, a laser, and a servo to actuate the laser will all communicate with a central server where the game logic and computation will take place. All peripherals will be attached with a WiFi-enabled Raspberry Pi Zero to stream video and communicate over the network which, along with the limited computation on the edge devices, will allow for easy deployment of the system. Key components of this project include time synchronization, person detection and identification, movement detection, and game logic in accordance with the series.

## 2. State of the Art & Its Limitations

The game is not currently implemented in real life through a machine, but it could be done so. Using existing algorithms of person detection and movement tracking, a very similar replica of the robot from the show could be built. There already exist various gaming devices that incorporate this idea of detecting a person to be a player in a game. Thus, the same methods can be used to perform what the robot doll did in the series. Regarding the shooting aspect, defense technology has evolved enough to precisely lock on targets that are desired. Such advancements are kept moreso a secret to the outside world to ensure it can’t be manipulated for harmful motives. However, it is very much possible in current practice. With a lot of time and research, all aspects of the game can be implemented with high success. The difficulty comes in when all aspects of various technologies have to be integrated together with limited resources. With no access or high understanding of complex algorithms, creating such machine to mimic the game becomes very complicated.

## 3. Novelty & Rationale

We will be implementing a novel method of server-based simultaneous multi-agent detection and tracking in real time. Using a single front-facing camera looking towards the players, we will develop a concurrent player tracking system to eliminate players and determine the game winners. We will implement complex tracking algorithms to handle occluded players. The peripherals will consist of a laser/actuator to “eliminate” players and cameras with no computation on the edge devices. This will make it much easier to synchronize and coordinate the different components of the game.

## 4. Potential Impact

In the technical sense, the implemented system can be seen as a naive version of the game. This means any further developments or implementations of Red Light, Green Light in the future can base its work from this and simply improve algorithms or methods used. Broadly, it encourages others to make this game and further it to perfection, as well as create other games from the series and convert its human aspects digitally. It also allows users to have fun with their friends by playing.

## 5. Challenges

Challenges root from reducing imperfections of our algorithms. A key challenge is to have person detection/tracking work even with occluded players in the scene, so that it doesn’t falsely mistrack someone. This can further bring obstacles when shooting a player with the laser. Distinguishing the players and pointing the laser at the correct person is highly difficult with many people overlapping. Another challenge is from streaming the video to the main system. Network latency as well as system limitations may cause more error during processing. Discrete computations may also create gaps between frames being processed and lead to undetected movement.

## 6. Requirements for Success

We want to make sure that we are able to identify players and distinguish them from one another even when occlusion occurs. We want to identify when players overlap so that they cannot avoid elimination by hiding behind someone else. We want to detect if a player has moved at all during a red light, locate them, and aim the laser at them to eliminate them without hitting other players. 

## 7. Metrics of Success

We would want to check that our person detection can maintain multiple bounding boxes throughout the red light periods and that the bounding boxes are preserved even when people overlap. For player elimination, we can measure the error from our intended target vs the actual one to determine our accuracy. We can also try to measure the delay between a player moving and their elimination.

## 8. Execution Plan

Joel will be working on person detection and motion detection. Anchal will be working on person identification/tracking and integration. Isha will be working on the video streaming, time synchronization, and integration.

## 9. Related Work

### 9.a. Papers
YOLOv4 for fast single-pass person detection: https://arxiv.org/abs/2004.10934 

YOLOv4-tiny for improved performance: https://arxiv.org/abs/2011.04244

DeepSORT for object tracking invariant to occlusion: https://arxiv.org/abs/1703.07402

### 9.b. Datasets

Yolov4-tiny weights/configuration used for person detection

Coco names file which includes object names for detection

MOT16 dataset for multi-object tracking used in DeepSORT

### 9.c. Software
Python, OpenCV for algorithm implementation

## 10. References

Bochkovskiy, A., Wang, C., & Liao, H.M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. ArXiv, abs/2004.10934. https://arxiv.org/abs/2004.10934 

https://github.com/AlexeyAB/darknet

https://github.com/pjreddie/darknet

Jiang, Z., Zhao, L., Li, S., & Jia, Y. (2020). Real-time object detection method based on improved YOLOv4-tiny. ArXiv, abs/2011.04244. https://arxiv.org/abs/2011.04244

Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. 2017 IEEE International Conference on Image Processing (ICIP), 3645-3649. https://arxiv.org/abs/1703.07402

https://github.com/nwojke/deep_sort

Milan, A., Leal-Taixé, L., Reid, I., Roth, S. & Schindler, K. MOT16: A Benchmark for Multi-Object Tracking. arXiv:1603.00831 [cs], 2016., (arXiv: 1603.00831). https://arxiv.org/abs/1603.00831
