# -*- coding: utf-8 -*-

import cv2
import numpy as np

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.generate_detections import create_box_encoder
from deep_sort.preprocessing import non_max_suppression

# Need to implement game system, laser, real time processing, improve overlaps
# Need initial time to calibrate

class Person:
    def __init__(self, img, origin, box, center_thres, box_thres ,number):
        # Initialize player traits
        self.origin = origin
        self.box = box
        self.center_thres = center_thres
        self.box_thres = box_thres
        self.number = number
        self.area = 1000000
        self.warning = 0
        self.out = 0
        self.current_box = box
        
    def update_current(self, box):
        self.current_box = box
        
    def change_in_center(self, center):
        # Return difference in center
        change_in_center = np.sqrt((center[1]-self.origin[1])**2+(center[0]-self.origin[0])**2)
        return change_in_center
    
    def check_movement(self, center, img, new_box, downsample_factor = 1/3):
        # Find change in center
        change_in_center = self.change_in_center(center)
        
        # Find change in bounding boxes
        old_x_len = self.box[2]
        old_y_len = self.box[3]
        new_x_len = new_box[2]
        new_y_len = new_box[3]
        change_in_box = abs(old_x_len-new_x_len)+abs(old_y_len-new_y_len)
        
        # Determine movement
        if self.out == 0:
            if change_in_center > self.center_thres*(old_x_len*old_y_len)/self.area or change_in_box > (self.box_thres*(old_x_len*old_y_len)/self.area):
                print('player %d : movement detected' % self.number)
                self.warning += 1
                if self.warning == 3:
                    print('player %d out' % self.number)
                    self.out = 1
            else:
                self.warning = max(0, self.warning - 1)
 
def check_state(red, net, tracker, encoder, classNames, img, thres, nms, start, players, rand, draw=True, end=0):
    # Detect objects
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    boxes = []
    
    if len(classIds) != 0:
        player_num = 1
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId]
            
            # For each person detected
            if className == 'person':
                boxes.append(box)
                
    # Track each person
    features = encoder(img, boxes)
    detections = [Detection(bbox, score, 'person', feature) for bbox, score, feature in zip(boxes, confs, features)]

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = non_max_suppression(boxs, classes, 1.0, scores)
    detections = [detections[i] for i in indices]       

    tracker.predict()
    tracker.update(detections)

    # update tracks
    player_num = 0
    for track in tracker.tracks:
        if not track.is_confirmed():# or track.time_since_update > 1:
            continue 
        
        box = track.to_tlbr()
        box = [int(b) for b in box]
        center = [box[1]+box[3]/2,box[0]+box[2]/2]
        
        # Initialize players
        if start == True:
            players.append(Person(img, center, box, 50, 50, track.track_id))
        
        # Run red light
        elif red:
            # Takes care of moving overlaps
            players[player_num].update_current(box)
            overlaps = 0
            for player in players:
                if player.number != track.track_id:
                    tbox = player.current_box
                    tcenter = [tbox[1]+tbox[3]/2,tbox[0]+tbox[2]/2]
                    if abs(center[1]-tcenter[1]) < 200:
                        overlaps = 1
 
            if overlaps != 1:
                players[player_num].check_movement(center, img, box, downsample_factor=1/4)
            player_num +=1
            
        # draw bbox 
        if cv2.__version__ == '4.5.1': # idek
            bbox = track.to_tlwh()
            cv2.rectangle(img, bbox, (0, 255, 0), 10)
        elif cv2.__version__ == '4.5.4-dev': 
            bbox = track.to_tlbr()
            cv2.rectangle(img, bbox[0:2].astype(int), bbox[2:].astype(int), (0, 255, 0), 2)
        cv2.putText(img, f'Person {track.track_id}', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    return img, players

def main():
    #Variables: box thresh, center thresh, net thresh and nms, input size, downsample factor, box total area
    
    # Upload net files
    configPath = '/Users/Joel Oh/Downloads/yolov4-tiny.cfg'
    weightsPath = '/Users/Joel Oh/Downloads/yolov4-tiny.weights'
    classFile = '/Users/Joel Oh/Downloads/coco.names.txt'
    classNames = []
    with open(classFile,"rt") as f:
        classNames = f.read().splitlines()
    
    # Create net
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320) #704,704
    net.setInputScale(1.0/ 255) #127.5 before
    #net.setInputMean((80, 80, 80)) #Determines overlapping
    net.setInputSwapRB(True)
    
    # Create tracker
    metric = NearestNeighborDistanceMetric("cosine", 0.5, None)
    tracker = Tracker(metric, max_age=100)
    encoder = create_box_encoder('/Users/Joel Oh/Downloads/mars-small128.pb', batch_size=1)

    # Upload video
    cap = cv2.VideoCapture('/Users/Joel Oh/Downloads/IMG_3086.MOV')
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    red_lights = [30, 300, 650, 980]
    #red_lights = [int(length*1/6)]
    green_lights = [120, 470, 850]
    frame_num = 0
    
    # Initialize variables
    red = False
    start = False
    players = []
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        if frame_num in red_lights:
            red = True
            start = True
            print('red')
            
        if frame_num in green_lights:
            red = False
            print('green')
            players = []
        
        result, players = check_state(red, net, tracker, encoder, classNames, img,0.65,0.4, start, players, rand = 0.0) # 0.45, 0.2 before, 0.1, 0.4
        start = False
            
        #print(objectInfo)
        img = cv2.resize(img, (720, 1080))
        cv2.imshow('img',img)
        cv2.waitKey(1)
        frame_num += 1
        
        if frame_num == length:
            players = []
            start = True
            red = True
            result, players = check_state(red, net, tracker, encoder, classNames, img,0.65,0.4, start, players, end=1, rand=0)

main()