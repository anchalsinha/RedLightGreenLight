from config import *
from utilities import *

from enum import Enum

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
        self.state = State.CONNECTING
    
    def connect(self):
        # get video streams
        self.state = State.GAME_START

    def start_game(self):
        # every player should be in a line, so start tracking and identification
        self.state = State.GREEN_LIGHT

    