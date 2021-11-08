import cv2
import os

from config import YOLOv4_TINY_MODEL_DIR

class PlayerTracker:
    def __init__(self):
        configPath = os.path.join(YOLOv4_TINY_MODEL_DIR, 'yolov4-tiny.cfg')
        weightsPath = os.path.join(YOLOv4_TINY_MODEL_DIR, 'yolov4-tiny.weights')
        classFile = os.path.join(YOLOv4_TINY_MODEL_DIR, 'coco.names.txt')
        with open(classFile,"rt") as f:
            self.classNames = f.read().splitlines()

        self.net = cv2.dnn_DetectionModel(weightsPath,configPath)
        self.net.setInputSize(320,320) #704,704
        self.net.setInputScale(1.0/ 255) #127.5 before
        #net.setInputMean((127.5, 127.5, 127.5)) #Determines overlapping
        self.net.setInputSwapRB(True)

    def detectPlayers(self, img, thres, nms, start, players, draw=True, objects=[]):
        # Detect objects
        classIds, confs, bbox = self.net.detect(img, confThreshold=thres, nmsThreshold=nms)
        if len(objects) == 0: 
            objects = self.classNames
        objectInfo =[]
        
        # Track people
        if len(classIds) != 0:
            player_num = 1
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                className = self.classNames[classId]
                
                # For each person detected
                if className == 'person':
                    
                    # some complicated determining who's who algorithm
                    # right now, just using who was closest to original place
                    
                    center = [box[1]+box[3]/2,box[0]+box[2]/2]
                    # Initialize people when start of red light
                    if start == True:
                        players.append(Person(img, center, box, 400, 400, player_num))
                        player_num += 1
                        
                    # Find original player by finding closest center match
                    changes = []
                    for player in players:
                        changes.append(player.change_in_center(center))
                    player = players[changes.index(min(changes))]
                    player.check_movement(center, img, box, error=0.3, downsample_factor=1/4)
                        
                    # Display info
                    objectInfo.append([box,className])
                    if (draw):
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,className.upper()+str(player.player_num()),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
        return img, objectInfo, players
    

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
                else:
                    print('player %d : no movement detected' % self.number)