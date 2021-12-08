# -*- coding: utf-8 -*-

import serial
import time
import numpy as np

Arduino = serial.Serial('COM4',9600)

def motor_angle(ang):   
    Arduino.write('0'.encode())
    Arduino.write('\n'.encode())
    for c in ang:
        Arduino.write(c.encode())      
    Arduino.write('\n'.encode())
        
def point_laser(players, x, thresh = 40):
    for player in players:
        if player.out == 1 and player.lasered == 0:
            player.lasered = 1
            player_center = (player.current_box[0]+player.current_box[2])/2
            player_center -= x/2
            angle = -(int(player_center/(x/2)*(thresh/2)+90)-180)
            print(int(angle))
            motor_angle(str(int(angle)))
            time.sleep(1.5)