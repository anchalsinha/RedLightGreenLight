import asyncio
import cv2
import time
import numpy as np
from enum import Enum
import time
import sys

from config import *
from utilities import *
from sound import *
from person import PlayerTracker
from sound import *
from arduino import *
import threading

class State(Enum):
    CONNECTING = 0
    CALIBRATION = 1
    GAME_START = 2
    GREEN_LIGHT = 3
    RED_LIGHT = 4
    RED_LIGHT_LASER = 5
    GAME_END = 6,
    SHOW_RESULTS = 7

sound = AudioSegment.from_file(file)
sound_length = 4

class Game:
    def __init__(self):
        # initialize any vars
        self.playerTracker = PlayerTracker()
        self.state = State.CONNECTING
        self.players = []
        self.outs = []

        self.game_timer = 0
        self.state_duration = 0
        self.state_timer = 0
        self.start = True
        self.sound_speed = 1
        self.x = 0
    
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
            self.game_timer += 1/fps + (time.time() - start_time)
            await asyncio.sleep(1/fps)

    def connect(self):
        # get video streams
        self.videoStream = cv2.VideoCapture(0)
        # self.videoStream = cv2.VideoCapture('udpsrc port=5200 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        self.state = State.GAME_START
        self.reset_state_timer((10,10)) 

    def start_game(self):
        # every player should be in a line, so start tracking and identification
        # 10 seconds from start to be ready
        current = time.time()
        t = threading.Thread(target=countdown, args=())
        t.start()
        while 1:
            ret, frame = self.videoStream.read()
            
            if (current+10-time.time()) >= 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(round(current+10-time.time(),2))
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                textX = int((frame.shape[1] - textsize[0]) / 2)
                textY = int((frame.shape[0] + textsize[1]) / 2)
                cv2.putText(frame, text, (textX, textY), font, 1, (255, 255, 255), 2)
                
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

            if time.time() > current+10:
                break

        while self.start == True:
            ret, frame = self.videoStream.read()
            frame, self.players, self.outs = self.playerTracker.detectPlayers(frame, 0.65, 0.4, self.start, False, self.players, self.outs)
            if len(self.players) != 0:
                self.start = False
                
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

    def reset_state_timer(self, duration_range):
        self.state_duration = np.random.uniform(duration_range[0], duration_range[1])
        self.state_timer = 0

    def green_light(self):
        ret, frame = self.videoStream.read()
        if not ret:
            return
        frame, self.players, self.outs = self.playerTracker.detectPlayers(frame, 0.65, 0.4, self.start, False, self.players, self.outs)
        
        if cv2.__version__ == '4.5.1': # anchal's version
            cv2.rectangle(frame, [0, 0, frame.shape[1],frame.shape[0]], (0, 255, 0), 25)
        elif cv2.__version__ == '4.5.4-dev' or cv2.__version__ == '4.5.4': # joel/isha's version
            cv2.rectangle(frame, [0,0], [frame.shape[1],frame.shape[0]], (0, 255, 0), 25)

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)


    def red_light(self):
        # Reset timer to fixed duration after laser is fired
        # self.reset_state_timer(RED_LIGHT_POST_DETECTION_DURATION) 
        ret, frame = self.videoStream.read()
        if not ret:
            return
        frame, self.players, self.outs = self.playerTracker.detectPlayers(frame, 0.65, 0.4, self.start, True, self.players, self.outs)
        
        if cv2.__version__ == '4.5.1': # anchal's version
            cv2.rectangle(frame, [0, 0, frame.shape[1],frame.shape[0]], (0, 0, 255), 25)
        elif cv2.__version__ == '4.5.4-dev' or cv2.__version__ == '4.5.4': # joel/isha's version
            cv2.rectangle(frame, [0,0], [frame.shape[1],frame.shape[0]], (0, 0, 255), 25)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

    def shoot_players(self):
        for player in self.players:
            if player.out == 1 and player.lasered == 0:
                while self.playerTracker.laser.is_alive():
                    continue
                point_laser(self.players)
                break
    
    def game_end(self):
        # Reset timer to fixed duration after laser is fired
        # self.reset_state_timer(RED_LIGHT_POST_DETECTION_DURATION) 
        ret, frame = self.videoStream.read()
        if not ret:
            return
        frame, self.players, self.outs = self.playerTracker.detectPlayers(frame, 0.65, 0.4, self.start, True, self.players, self.outs, end=1)
        
        if cv2.__version__ == '4.5.1': # anchal's version
            cv2.rectangle(frame, [0, 0, frame.shape[1],frame.shape[0]], (255, 0, 0), 25)
        elif cv2.__version__ == '4.5.4-dev' or cv2.__version__ == '4.5.4': # joel/isha's version
            cv2.rectangle(frame, [0,0], [frame.shape[1],frame.shape[0]], (255, 0, 0), 25)

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

        if all([(player.out == 1 and player.lasered == 1) or player.out == 0 for player in self.players]): # all players lasered
            self.state = State.SHOW_RESULTS


    def manage_state(self):
        if self.state_timer > self.state_duration:
            if self.game_timer >= GAME_DURATION and self.state != State.SHOW_RESULTS:
                self.state = State.GAME_END
            elif self.state == State.GREEN_LIGHT:
                self.state = State.RED_LIGHT
                self.reset_state_timer(RED_LIGHT_DURATION_RANGE)
                print("Current State: RED LIGHT")
            elif self.state == State.RED_LIGHT or self.state == State.GAME_START:
                self.shoot_players()
                self.state = State.GREEN_LIGHT
                self.reset_state_timer(GREEN_LIGHT_DURATION_RANGE)
                print("Current State: GREEN LIGHT")
                self.sound_speed = dur()/self.state_duration
                t = threading.Thread(target=play_sound, args=(self.sound_speed,))
                t.start()

        if self.state == State.CONNECTING:
            self.connect()
        
        elif self.state == State.GAME_START:
            self.start_game()
        elif self.state == State.GREEN_LIGHT:
            self.green_light()
        elif self.state == State.RED_LIGHT:
            self.red_light()
        elif self.state == State.GAME_END:
            self.game_end()
        elif self.state == State.SHOW_RESULTS:
            print('Showing Results')
            losers = ""
            winners = ""
            for player in self.players:
                if player.out == 0:
                    winners += " Player %d " % (player.number)
                else:
                    losers += " Player %d " % (player.number)
                    
            print("Winners:"+winners)
            print("Losers:"+losers)
            sys.exit(0)

