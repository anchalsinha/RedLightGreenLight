import cv2
import os
import numpy as np
from collections import namedtuple
from sewar.full_ref import ssim

# from deep_sort.tracker import Tracker
# from deep_sort.nn_matching import NearestNeighborDistanceMetric
# from deep_sort.detection import Detection
# from deep_sort.generate_detections import create_box_encoder
# from deep_sort.preprocessing import non_max_suppression
from yolox.tracker.byte_tracker import BYTETracker
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
        self.tracker = BYTETracker(args)
        self.players = []
        self.eliminationQueue = []

    def detectPlayers(self, frame, conf_threshold, nms_threshold, startRed, redLight):
        if startRed:
            self.players = []
            self.eliminationQueue = []
        # Detect objects
        class_ids, confidences, bboxes = self.net.detect(frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
        detections = []

        boxes = []
        confs = []

        if len(class_ids) == 0:
            return frame, None

        player_num = 1

        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bboxes):
            className = self.classNames[class_id]
            # For each person detected
            if className == 'person':
                boxes.append(box)
                confs.append(confidence)

                center = [box[1]+box[3]/2,box[0]+box[2]/2]
                if startRed:
                    self.players.append(Person(frame, center, box, 400, 400, player_num))
                    player_num += 1

                if redLight:
                    changes = []
                    for player in self.players:
                        changes.append(player.change_in_center(center))

                    if changes:
                        player = self.players[changes.index(min(changes))]
                        res = player.check_movement(center, frame, box, error=0.3, downsample_factor=1/4)
                        if res and player not in self.eliminationQueue:
                            self.eliminationQueue.append(player)
                            print("eliminate")
        
        # features = self.encoder(frame, boxes)
        # detections = [Detection(bbox, score, 'person', feature) for bbox, score, feature in zip(boxes, confs, features)]
        detections = np.array([[bbox[0], bbox[1], bbox[2], bbox[3], score] for bbox, score in zip(boxes, confs)])

        # run non-maxima supression
        # boxs = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # classes = np.array([d.class_name for d in detections])
        # indices = non_max_suppression(boxs, classes, 1.0, scores)
        # detections = [detections[i] for i in indices]       

        # self.tracker.predict()
        tracks = self.tracker.update(detections, [640, 480], [640, 480])

        # update tracks
        for track in tracks:
            # draw bbox 
            if cv2.__version__ == '4.5.1': # idek
                bbox = track.tlwh
                cv2.rectangle(frame, bbox, (0, 255, 0), 10)
            elif cv2.__version__ == '4.5.4-dev' or cv2.__version__ == '4.5.4': 
                bbox = track.tlbr
                cv2.rectangle(frame, bbox[0:2].astype(int), bbox[2:].astype(int), (0, 255, 0), 10)
            cv2.putText(frame, f'ID: {track.track_id}', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        return frame, detections
    

class Person:
    def __init__(self, img, origin, box, center_thres, box_thres ,number):
        # Initialize player traits
        self.origin = origin
        self.box = box
        self.center_thres = center_thres
        self.box_thres = box_thres
        self.number = number
        self.area = 1000000
        self.person_old_box = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    
    def player_num(self):
        # Return player number
        return self.number
        
    def change_in_center(self, center):
        # Return difference in center
        change_in_center = np.sqrt((center[1]-self.origin[1])**2+(center[0]-self.origin[0])**2)
        return change_in_center
    
    def check_movement(self, center, img, new_box, error, downsample_factor = 1/3):
        # Find change in center
        change_in_center = self.change_in_center(center)
        
        # Find change in bounding boxes
        old_x_len = self.box[2]
        old_y_len = self.box[3]
        new_x_len = new_box[2]
        new_y_len = new_box[3]
        change_in_box = abs(old_x_len-new_x_len)+abs(old_y_len-new_y_len)
        
        # Determine movement
        if self.number == 1:
            if change_in_center > self.center_thres*(old_x_len*old_y_len)/self.area or change_in_box > (self.box_thres*(old_x_len*old_y_len)/self.area):
                print('player %d : movement detected' % self.number)
                return True
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                person_new_box = gray[new_box[1]:new_box[1]+new_box[3],new_box[0]:new_box[0]+new_box[2]]
    
                
                downsample_old = cv2.resize(self.person_old_box, None, fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_AREA)
                downsample_new = cv2.resize(person_new_box, None, fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_AREA)
                
                m = [downsample_old.shape[0], downsample_new.shape[0]].index(min(downsample_old.shape[0], downsample_new.shape[0]))
                n = [downsample_old.shape[1], downsample_new.shape[1]].index(min(downsample_old.shape[1], downsample_new.shape[1]))
                y = [downsample_old.shape[0], downsample_new.shape[0]][m]
                x = [downsample_old.shape[1], downsample_new.shape[1]][n]
                y = int(y/2)
                x = int(x/2)
                downsample_old = downsample_old[int(downsample_old.shape[0]/2)-y:int(downsample_old.shape[0]/2)+y,
                                                int(downsample_old.shape[1]/2)-x:int(downsample_old.shape[1]/2)+x]
                downsample_new = downsample_new[int(downsample_new.shape[0]/2)-y:int(downsample_new.shape[0]/2)+y,
                                                int(downsample_new.shape[1]/2)-x:int(downsample_new.shape[1]/2)+x]
                
                #print("MSE: ", mse(downsample_new,downsample_old))
                #print("RMSE: ", rmse(downsample_new, downsample_old))
                #print("PSNR: ", psnr(downsample_new, downsample_old))
                #print("SSIM: ", ssim(downsample_new, downsample_old))
                #print("UQI: ", uqi(downsample_new, downsample_old))
                #print("ERGAS: ", ergas(downsample_new, downsample_old))
                #err = scc(downsample_new, downsample_old)
                #print("RASE: ", rase(downsample_new, downsample_old))
                #print("SAM: ", sam(downsample_new, downsample_old))
                err = np.mean(ssim(downsample_new, downsample_old))
                    
                if err < error:
                    print('player %d : movement detected' % self.number)
                    return True
                else:
                    print('player %d : no movement detected' % self.number)
        return False