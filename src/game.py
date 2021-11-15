import asyncio
import cv2
from enum import Enum

from config import *
from utilities import *
from person import PlayerTracker

class State(Enum):
    CONNECTING = 0
    CALIBRATION = 1
    GAME_START = 2
    GREEN_LIGHT = 3
    RED_LIGHT = 4
    RED_LIGHT_LASER = 5
    GAME_END = 6


class Game:
    def __init__(self):
        # initialize any vars
        self.playerTracker = PlayerTracker()
        self.state = State.CONNECTING
    
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
            self.manage_state()
            await asyncio.sleep(1/fps)

    def connect(self):
        # get video streams
        # self.videoStream = cv2.VideoCapture(0)
        self.videoStream = cv2.VideoCapture('udpsrc port=5200 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        self.state = State.GAME_START

    def start_game(self):
        # every player should be in a line, so start tracking and identification
        self.state = State.GREEN_LIGHT

    def green_light(self):
        ret, frame = self.videoStream.read()
        if not ret:
            return
        frame, detections = self.playerTracker.detectPlayers(frame, 0.65, 0.4)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

    def manage_state(self):
        if self.state == State.CONNECTING:
            self.connect()
        elif self.state == State.GAME_START:
            self.start_game()
        elif self.state == State.GREEN_LIGHT:
            self.green_light()
        elif self.state == State.RED_LIGHT:
            pass
