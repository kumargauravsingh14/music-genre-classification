
# coding: utf-8

# In[1]:


# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Extracting music and features
'''
Dataset
We use GTZAN genre collection dataset for classification. 

The dataset consists of 10 genres i.e

Blues
Classical
Country
Disco
Hiphop
Jazz
Metal
Pop
Reggae
Rock
Each genre contains 100 songs. Total dataset: 1000 songs
'''


# In[3]:


# Extracting the Spectrogram for every Audio


# In[4]:



cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'./MIR/genres/{g}'):
        songname = f'./MIR/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()



# In[5]:


'''
Extracting features from Spectrogram
We will extract

Mel-frequency cepstral coefficients (MFCC)(20 in number)
Spectral Centroid,
Zero Crossing Rate
Chroma Frequencies
Spectral Roll-off.
'''


# In[6]:


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


# In[7]:


# Writing data to csv file


# In[9]:


file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'./MIR/genres/{g}'):
        songname = f'./MIR/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


# In[10]:


# Analysing the Data in Pandas


# In[11]:


data = pd.read_csv('data.csv')
data.head()


# In[12]:


data.shape


# In[13]:


# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)


# In[14]:


# Encoding the Labels


# In[15]:


genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


# In[16]:


# Scaling the Feature columns


# In[17]:


scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))


# In[18]:


# Dividing data into training and Testing set


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[20]:


len(y_train)


# In[21]:


len(y_test)


# In[22]:


X_train[10]


# In[23]:


# Classification with Keras
# Building our Network


# In[24]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))


# In[25]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[26]:


history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)


# In[27]:


test_loss, test_acc = model.evaluate(X_test,y_test)


# In[28]:


print('test_acc: ',test_acc)


# In[29]:


# Tes accuracy is less than training data accuracy. This hints at Overfitting.


# In[30]:


# Validating our approach
# Let's set apart 200 samples in our training data to use as a validation set:


# In[31]:


x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]


# In[32]:


# Now let's train our network for 20 epochs:


# In[33]:


model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=30,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)


# In[34]:


results


# In[35]:


# Predictions on Test Data


# In[36]:


predictions = model.predict(X_test)


# In[37]:


predictions[0].shape


# In[38]:


np.sum(predictions[0])


# In[39]:


np.argmax(predictions[0])

