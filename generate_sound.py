import pyaudio
import numpy as np
import time


volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 0.5   # in seconds, may be float
f_one = 18000.0  # sine frequency, Hz, may be float
f_zero = 40.0   # sine frequency, Hz, may be float

import threading




def close_stream(pobj,stream):
    stream.stop_stream()
    stream.close()
    pobj.terminate()

def sound_generator(pobj,stream,bin_data):
    #print(bin_data)
    for bit in bin_data:
        if bit == '1':
            # generate samples, note conversion to float32 array
            samples = (np.sin(2*np.pi*np.arange(fs*duration)*f_one/fs)).astype(np.float32)
            # play. May repeat with different volume values (if done interactively) 
            stream.write(volume*samples)
        if bit == '0':
            # generate samples, note conversion to float32 array
            samples = (np.sin(2*np.pi*np.arange(fs*duration)*f_zero/fs)).astype(np.float32)
            # play. May repeat with different volume values (if done interactively) 
            stream.write(volume*samples)

