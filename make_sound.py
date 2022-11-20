#import sounddevice as sd
import wave
import numpy as np

from playsound import playsound
while True:
    playsound("loop1.wav")


#wf = wave.open("loop1.wav")
#fs = wf.getframerate()
#data = wf.readframes(wf.getnframes())
#data = np.frombuffer(data, dtype='int16')

#sd.play(data, fs)
#status = sd.wait()