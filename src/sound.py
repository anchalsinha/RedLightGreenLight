# -*- coding: utf-8 -*-

import time
import numpy as np
import wave
import contextlib
from pydub import AudioSegment
from pydub.playback import play
#install pyaudio

file = './sounds/voice.wav'
sound = AudioSegment.from_file(file)
count = './sounds/countdown.wav'
count = AudioSegment.from_file(count)

def countdown():
    play(count)

def dur(fi=file):
    with contextlib.closing(wave.open(file,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def play_sound(speed=1.0):
    changed_sound = speed_change(sound, speed)
    play(changed_sound)  

# t_end = time.time()+15 #game runs for 15 seconds
# while time.time() < t_end:
#     speed = np.random.uniform(0.7, 1.5)
#     # changed_sound = speed_change(speed)
#     # play(changed_sound)
#     play_sound(speed)
    #red light begins
    #read in frame and run processing
    

    