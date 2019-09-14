import os
import librosa 
import IPython.display as ipd
import sounddevice as sd
import soundfile as sf

import numpy as np
import warnings

from keras.models import load_model
model=load_model('model.hdf5')
def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()
#x_value=loadList("x_value.npy")
#y_value=loadList("y_value.npy")
#x_train=loadList("x_train.npy")
#sy_train=loadList("y_train.npy")
classes=loadList("classes.npy")
print("done")
def predict(audio):
	global classes
	prediction=model.predict(audio.reshape(1,8000,1))
	index=np.argmax(prediction[0])
	return classes[index]
samplerate = 16000  
duration = 1 
filename = 'yes.wav'

samplerate = 16000  
duration = 1 # seconds
filename = 'yes.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
print(mydata)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)

os.listdir('../sample_submission/train/audio/on')
filepath='../sample_submission/train/audio/on/'

samples, sample_rate = librosa.load(filepath  + 'yes.wav', sr = 16000)
print(samples)

samples = librosa.resample(samples, sample_rate, 8000)
#ipd.Audio(samples,rate=8000)	


predict(samples)
print("Text:",predict(samples))