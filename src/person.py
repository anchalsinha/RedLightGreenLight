import cv2
import os
import numpy as np
from collections import namedtuple

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.generate_detections import create_box_encoder
from deep_sort.preprocessing import non_max_suppression
#from yolox.tracker.byte_tracker import BYTETracker
from config import YOLOv4_TINY_MODEL_DIR

class PlayerTracker:
    def __init__(self, track_thresh=0.65, match_thresh=0.9, track_buffer=30):
        configPath = os.path.join(YOLOv4_TINY_MODEL_DIR, 'yolov4-tiny.cfg')
        weightsPath = os.path.join(YOLOv4_TINY_MODEL_DIR, 'yolov4-tiny.weights')
        classFile = os.path.join(YOLOv4_TINY_MODEL_DIR, 'coco.names.txt')
        marsPath = os.path.join(YOLOv4_TINY_MODEL_DIR, 'mars-small128.pb')
        with open(classFile,"rt") as f:
            self.classNames = f.read().splitlines()

        self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
        self.net.setInputSize(320,320) #704,704
        self.net.setInputScale(1.0/ 255) #127.5 before
        #net.setInputMean((127.5, 127.5, 127.5)) #Determines overlapping
        self.net.setInputSwapRB(True)

        argsObject = namedtuple('args', 'track_thresh match_thresh track_buffer mot20')
        args = argsObject(track_thresh, match_thresh, track_buffer, False)
        #self.tracker = BYTETracker(args)
        metric = NearestNeighborDistanceMetric("cosine", 0.5, None)
        self.tracker = Tracker(metric, max_age=100)
        self.encoder = create_box_encoder(os.path.join(YOLOv4_TINY_MODEL_DIR, 'mars-small128.pb'), batch_size=1)
        self.players = []
        self.eliminationQueue = []

    def detectPlayers(self, frame, conf_threshold, nms_threshold, start, redLight, players, outs, end=0):
        # Detect objects
        classIds, confs, bbox = self.net.detect(frame,confThreshold=conf_threshold,nmsThreshold=nms_threshold)
        boxes = []
        
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                className = self.classNames[classId]
                
                # For each person detected
                if className == 'person':
                    boxes.append(box)
                    
        # Track each person
        features = self.encoder(frame, boxes)
        detections = [Detection(bbox, score, 'person', feature) for bbox, score, feature in zip(boxes, confs, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = non_max_suppression(boxs, classes, 1.0, scores)
        detections = [detections[i] for i in indices]       

        self.tracker.predict()
        self.tracker.update(detections)

        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed():# or track.time_since_update > 1:
                continue 
            
            if end == 1:
                players[track.track_id-1].out = 1
            
            box = track.to_tlbr()
            box = [int(b) for b in box]
            center = [box[1]+box[3]/2,box[0]+box[2]/2]
            
            # Initialize players
            if start:
                players.append(Person(frame, center, box, 400, 400, track.track_id))
            
            # Run red light
            elif redLight:
                # Takes care of moving overlaps
                players[track.track_id-1].update_current(box)
                overlaps = 0
                for player in players:
                    if player.number != track.track_id:
                        tbox = player.current_box
                        tcenter = [tbox[1]+tbox[3]/2,tbox[0]+tbox[2]/2]
                        if abs(center[1]-tcenter[1]) < 200*(player.box[2]*player.box[3])/player.area:
                            overlaps = 1
     
                players[track.track_id-1].check_movement(center, frame, box, overlaps)
                
            if start == False and players[track.track_id-1].out == 0:
                # draw bbox 
                bbox = track.to_tlbr()
                cv2.rectangle(frame, bbox[0:2].astype(int), bbox[2:].astype(int), (0, 255, 0), 2)
                cv2.putText(frame, f'Player {track.track_id}', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
        for player in players:
            if player.number not in outs and player.out == 1:
                print('player %d out' % player.number)
                outs.append(player.number)
            
        return frame, players, outs
    
class Person:
    def __init__(self, img, origin, box, center_thres, box_thres, number):
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
    
    def check_movement(self, center, img, new_box, overlaps):
        # Find change in center
        change_in_center = self.change_in_center(center)
        
        # Find change in bounding boxes
        old_x_len = self.box[2]
        old_y_len = self.box[3]
        new_x_len = new_box[2]
        new_y_len = new_box[3]
        change_in_box = abs(old_x_len-new_x_len)+abs(old_y_len-new_y_len)
        
        # Determine movement
        '''
        criterion = False

        if overlaps == 0:
            if change_in_center > self.center_thres*(old_x_len*old_y_len)/self.area or change_in_box > (self.box_thres*(old_x_len*old_y_len)/self.area):
                criterion = True
        
        else:
            if change_in_center > self.center_thres*1.5*(old_x_len*old_y_len)/self.area:
                criterion = True
        '''
        factor = 1
        
        if overlaps == 1:
            factor = 1.5
        
        if self.out == 0:
            if change_in_center > self.center_thres*factor*(old_x_len*old_y_len)/self.area or change_in_box > (self.box_thres*factor*(old_x_len*old_y_len)/self.area):
                print('player %d : movement detected' % self.number)
                self.warning += 1
                if self.warning == 3:
                    self.out = 1
            else:
                self.warning = max(0, self.warning - 1)