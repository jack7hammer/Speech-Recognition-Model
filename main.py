import os
import librosa 
import IPython.display as ipd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")
def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()

warnings.filterwarnings("ignore")

path='../sample_submission/train/audio/'
samples, sample_rate = librosa.load(path+'yes/0a7c2a8d_nohash_0.wav', sr = 8000)
print(samples,sample_rate)
#ipd.audio(samples,rate=8000)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)


labels=os.listdir(path)

Allwaves=[]
AllLabels=[]
for label in labels:
	waves= [ f for f in os.listdir(path+'/'+label) if f.endswith('.wav')]
	for wav in waves:
		samples,sample_rate=librosa.load(path+label+'/'+wav,sr=16000)
		samples= librosa.resample(samples,sample_rate,8000)
		if(len(samples)==8000):
			Allwaves.append(samples)
			AllLabels.append(label)

#print(Allwaves,AllLabels)
encoder=LabelEncoder()
temp=encoder.fit_transform(AllLabels)
classes=list(encoder.classes_)
saveList(classes,"classes.npy")
temp=np_utils.to_categorical(temp,num_classes=len(labels))
Allwaves=np.array(Allwaves).reshape(-1,8000,1)

#x_train,x_value,y_train,y_value=train_test_split(np.array(Allwaves),np.array(temp),stratify=temp,test_size=0.25,random_state=777,shuffle=True)
#saveList(x_value,"x_value.npy")
#saveList(y_value,"y_value.npy")
#saveList(x_train,"x_train.npy")
#saveList(y_train,"y_train.npy")


#MODEL 

K.clear_session()

inputs=Input(shape=(8000,1))
conv= Conv1D(8,13,padding='valid',activation='relu',strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv= Conv1D(16,11,padding='valid',activation='relu',strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv= Conv1D(32,9,padding='valid',activation='relu',strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv= Conv1D(64,7,padding='valid',activation='relu',strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv=Flatten()(conv)

conv=Dense(256,activation='relu')(conv)
conv=Dropout(0.3)(conv)

conv=Dense(128,activation='relu')(conv)
conv=Dropout(0.3)(conv)

outputs=Dense(len(labels),activation='softmax')(conv)
model=Model(inputs,outputs)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


history=model.fit(x_train, y_train ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_value,y_value))



