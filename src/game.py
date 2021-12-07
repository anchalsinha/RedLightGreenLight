import asyncio
import cv2
import time
import numpy as np
from enum import Enum

from config import *
from utilities import *
from sound import *
from person import PlayerTracker
import threading

class State(Enum):
    CONNECTING = 0
    CALIBRATION = 1
    GAME_START = 2
    GREEN_LIGHT = 3
    RED_LIGHT = 4
    RED_LIGHT_LASER = 5
    GAME_END = 6

sound = AudioSegment.from_file(file)
sound_length = 4

class Game:
    def __init__(self):
        # initialize any vars
        self.playerTracker = PlayerTracker()
        self.state = State.CONNECTING

        self.state_duration = 0
        self.state_timer = 0
        self.startRed = False
        self.sound_speed = 1
        
    
    def run(self):
        loop = asyncio.get_event_loop()
        # loop.call_later(5, stop) # to stop the loop
        task = loop.create_task(self.timer(30))
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            task.cancel()

    async def timer(self, fps):
        while True:
            # control periodic tasks
            start_time = time.time()
            self.manage_state()
            self.state_timer += 1/fps + (time.time() - start_time)
            await asyncio.sleep(1/fps)

    def connect(self):
        # get video streams
        self.videoStream = cv2.VideoCapture(0)
        # self.videoStream = cv2.VideoCapture('udpsrc port=5200 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        self.state = State.GAME_START

    def start_game(self):
        # every player should be in a line, so start tracking and identification
        self.state = State.GREEN_LIGHT
        self.reset_state_timer(GREEN_LIGHT_DURATION_RANGE)
        print("Starting game")
        print("Current State: GREEN LIGHT")
        # play_sound(self.sound_speed)
        t = threading.Thread(target=play_sound, args=(self.sound_speed,))
        t.start()

    def reset_state_timer(self, duration_range):
        self.state_duration = np.random.uniform(duration_range[0], duration_range[1])
        self.state_timer = 0

    def green_light(self):
        ret, frame = self.videoStream.read()
        if not ret:
            return
        frame, detections = self.playerTracker.detectPlayers(frame, 0.65, 0.4, False, False)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)


    def red_light(self):
        '''
        TODO: 
         - Randomly select duration and make it minimum duration of red light state
         - Check movement continuously
         - If movement detected, fire the laser (probably queue up the shooting)
         - Add fixed delay after the shooting
        '''

        # Reset timer to fixed duration after laser is fired
        # self.reset_state_timer(RED_LIGHT_POST_DETECTION_DURATION) 
        ret, frame = self.videoStream.read()
        if not ret:
            return
        frame, detections = self.playerTracker.detectPlayers(frame, 0.65, 0.4, self.startRed, True)
        self.startRed = False
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

    def manage_state(self):
        if self.state_timer > self.state_duration:
            if self.state == State.GREEN_LIGHT:
                self.state = State.RED_LIGHT
                print(f'Actual state duration: {self.state_timer}, target: {self.state_duration}')
                self.reset_state_timer(RED_LIGHT_DURATION_RANGE)
                self.startRed = True
                t = threading.Thread(target=play_sound, args=(self.sound_speed,))
                t.start()
                print("Current State: RED LIGHT")
            elif self.state == State.RED_LIGHT:
                self.state = State.GREEN_LIGHT
                print(f'Actual state duration: {self.state_timer}, target: {self.state_duration}')
                self.reset_state_timer(GREEN_LIGHT_DURATION_RANGE)
                self.sound_speed = self.sound_speed * 1.1
                t = threading.Thread(target=play_sound, args=(self.sound_speed,))
                t.start()
                # play_sound(self.sound_speed)
                print("Current State: GREEN LIGHT")

        if self.state == State.CONNECTING:
            self.connect()
        
        elif self.state == State.GAME_START:
            self.start_game()
        elif self.state == State.GREEN_LIGHT:
            self.green_light()
        elif self.state == State.RED_LIGHT:
            self.red_light()
            # pass
