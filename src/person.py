import cv2
import os

class PlayerTracker:
    def __init__(self, model_dir):
        configPath = os.path.join(model_dir, 'yolov4.cfg')
        weightsPath = os.path.join(model_dir, 'yolov4.weights')
        classFile = os.path.join(model_dir, 'coco.names.txt')
        with open(classFile,"rt") as f:
            self.classNames = f.read().splitlines()

        self.net = cv2.dnn_DetectionModel(weightsPath,configPath)
        self.net.setInputSize(320,320) #704,704
        self.net.setInputScale(1.0/ 255) #127.5 before
        #net.setInputMean((127.5, 127.5, 127.5)) #Determines overlapping
        self.net.setInputSwapRB(True)

    def detectPlayers(self, img, thres, nms, start, players, draw=True, objects=[]):
        player_num = 1
        classIds, confs, bbox = self.net.detect(img,confThreshold=thres,nmsThreshold=nms)
        
        if len(objects) == 0: objects = self.classNames
        objectInfo =[]
        
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                className = self.classNames[classId]
                #if className in objects: 
                if className == 'person':
                    
                    # some complicated determining who's who algorithm
                    # right now, just using who was closest to original place
                    
                    center = [(box[1]+box[0])/2, (box[3]+box[2])/2]
                    if start == True:
                        players.append(Person(center, box, 50, 100, player_num))
                        player_num += 1
                        
                    changes = []
                    for player in players:
                        changes.append(player.change_in_center(center))
                    player = players[changes.index(min(changes))]
                    player.check_movement(center, box)
                        
                    objectInfo.append([box,className])
                    if (draw):
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,className.upper()+str(player.player_num()),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
        return img, objectInfo, players
    
class Person:
    def __init__(self, origin, box, center_thres, box_thres ,number):
        self.origin = origin
        self.box = box
        self.center_thres = center_thres
        self.box_thres = box_thres
        self.number = number
    
    def player_num(self):
        return self.number
        
    def change_in_center(self, center):
        change_in_center = np.sqrt((center[1]-self.origin[1])**2+(center[0]-self.origin[0])**2)
        return change_in_center
    
    def check_movement(self, center, new_box):
        # make threshold proportional to bounding box size
        change_in_center = np.sqrt((center[1]-self.origin[1])**2+(center[0]-self.origin[0])**2)
        old_x_len = self.box[1]-self.box[0]
        old_y_len = self.box[3]-self.box[2]
        new_x_len = new_box[1]-new_box[0]
        new_y_len = new_box[3]-new_box[2]
        change_in_box = abs(old_x_len-new_x_len)+abs(old_y_len-new_y_len)
        if change_in_center > self.center_thres or change_in_box > self.box_thres:
            print('player %d : movement detected' % self.number)
        else:
            print('player %d : no movement detected' % self.number)