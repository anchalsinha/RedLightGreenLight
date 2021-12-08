# -*- coding: utf-8 -*-

import serial
import time
import numpy as np

Arduino = serial.Serial('COM4',9600)

def motor_angle(ang):
    while 1:       
        for c in ang:
            Arduino.write(c.encode())      
        Arduino.write('\n'.encode())
        
def point_laser(players):
    for player in players:
        if player.out == 1 and player.lasered == 0:
            player.lasered = 1
            # calculate angle
            angle = np.random.randint(0,180)
            motor_angle(str(angle))
            time.sleep(1)