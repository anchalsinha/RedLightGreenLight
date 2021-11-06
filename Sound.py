# -*- coding: utf-8 -*-

import time
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
#install pyaudio

file = '/Users/Joel Oh/Downloads/voice.wav'
sound = AudioSegment.from_file(file)

def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

t_end = time.time()+15 #game runs for 15 seconds
while time.time() < t_end:
    speed = np.random.uniform(0.7, 1.5)
    changed_sound = speed_change(sound, speed)
    play(changed_sound)
    
    #red light begins
    #read in frame and run processing
    

    