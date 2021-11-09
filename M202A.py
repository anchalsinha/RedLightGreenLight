# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

# Need to implement person tracking - who's who, game system, laser, real time processing

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
        self.person_color = np.mean(cv2.cvtColor(img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]], cv2.COLOR_RGB2HSV))
        self.warning = 0
        self.out = 0
        self.tracker = cv2.TrackerMIL_create()
        self.tracker.init(img, box)
    
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
        if self.out == 0:
            if change_in_center > self.center_thres*(old_x_len*old_y_len)/self.area or change_in_box > (self.box_thres*(old_x_len*old_y_len)/self.area):
                print('player %d : movement detected' % self.number)
                self.warning += 1
                if self.warning == 3:
                    print('player %d out' % self.number)
                    self.out = 1
            else:
                self.warning = max(0, self.warning - 1)
 
def red_light(net, classNames, img, thres, nms, start, players, rand, draw=True, objects=[], end=0):
    # Detect objects
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: 
        objects = classNames
    objectInfo =[]
    
    # Detect movement
    closest = []
    centers = []
    boxes = []
    confidences = []
    
    if len(classIds) != 0:
        player_num = 1
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId]
            
            # For each person detected
            if className == 'person':
                
                center = [box[1]+box[3]/2,box[0]+box[2]/2]
                # Initialize people when start of red light
                if start == True:
                    players.append(Person(img, center, box, 400, 400, player_num))
                    player_num += 1
                    
                else:
                    changes = []
                    for player in players:
                        changes.append(player.change_in_center(center))
                    loc = [0 for i in range(len(players))]
                    loc[changes.index(min(changes))] = 1
                    closest.append(loc)
                    centers.append(center)
                    boxes.append(box)
                    confidences.append(confidence)
                    
    if start == False:
        ###################################################################
        if len(closest) != 0 and (len(closest) != len(players) or abs(np.linalg.det(closest)) != 1):
            uncertains = []
            for i in range(len(closest[0])):
                if np.sum(np.array(closest)[:,i]) != 1:
                    uncertains.append(i)
                    
            new_centers = []
            unknowns = []
            for uncertain in uncertains:
                if np.sum(np.array(closest)[:,uncertain]) != 0:
                    inds = [i for i, e in enumerate(list(np.array(closest)[:,uncertain])) if e == 1]
                    for ind in inds:
                        unknowns.append(ind)
                
                ok, new_box = players[uncertain].tracker.update(img)
                players[uncertain].box = new_box
                new_centers.append([new_box[1]+new_box[3]/2,new_box[0]+new_box[2]/2])
                
            results = []
            for unknown in unknowns:
                changes = []
                for new_center in new_centers:
                    changes.append(np.sqrt((new_center[1]-centers[unknown][1])**2+(new_center[0]-centers[unknown][0])**2))
                results.append(uncertains[changes.index(min(changes))])
                
                '''
            for result in results:
                r = all(element == result for element in results)
                if r == True:
                    inds = [i for i, e in enumerate(results) if e == result]
                
            
            '''
                players[uncertains[changes.index(min(changes))]].check_movement(centers[unknown], img, boxes[unknown], error=0.3, downsample_factor=1/4)
            
                # Display info
                objectInfo.append([boxes[unknown],'PERSON'])
                if (draw):
                    cv2.rectangle(img,boxes[unknown],color=(0,255,0),thickness=2)
                    cv2.putText(img,"PERSON"+str(players[uncertains[changes.index(min(changes))]].player_num()),(boxes[unknown][0]+10,boxes[unknown][1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidences[unknown]*100,2)),(boxes[unknown][0]+200,boxes[unknown][1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    
        #################################################################
        else:
            for i in range(len(closest)):
                ind = closest[i].index(1)
                players[ind].check_movement(centers[i], img, boxes[i], error=0.3, downsample_factor=1/4)
                #randomize some updates
                if np.random.random() < rand:
                    players[ind].tracker.update(img)
                
                # Display info
                objectInfo.append([boxes[i],'PERSON'])
                if (draw):
                    cv2.rectangle(img,boxes[i],color=(0,255,0),thickness=2)
                    cv2.putText(img,"PERSON"+str(players[ind].player_num()),(boxes[i][0]+10,boxes[i][1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidences[i]*100,2)),(boxes[i][0]+200,boxes[i][1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
        if end == 1:
            for player in players:
                print('player %d out' % player.number)
    
    return img,objectInfo, players

def main():
    #Variables: box thresh, center thresh, net thresh and nms, input size, scc error, downsample factor, box total area
    
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

    # Upload video
    cap = cv2.VideoCapture('/Users/Joel Oh/Downloads/IMG_3111.MOV')
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #red_lights = [30, 300, 650, 980]
    red_lights = [int(length*1/6)]
    #green_lights = [120, 470, 850]
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
            
        if frame_num-5000 in red_lights:# in green_lights:
            red = False
            print('green')
            players = []
        
        if red:
            result, objectInfo, players = red_light(net, classNames, img,0.65,0.4, start, players, rand = 0.0) # 0.45, 0.2 before, 0.1, 0.4
            start = False
            
        #print(objectInfo)
        img = cv2.resize(img, (720, 1080))
        cv2.imshow('img',img)
        cv2.waitKey(1)
        frame_num += 1
        
        if frame_num == length:
            players = []
            start = True
            result, objectInfo, players = red_light(net, classNames, img,0.65,0.4, start, players, end=1, rand=0)

main()