# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:26:38 2018

@author: sugho
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 15:49:31 2018

@author: sugho
"""

import os
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

samples = []
with open('./MyDrivingData/combined_normalized_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# Split Training and Validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

# Generator to supply images when needed for training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=30)
validation_generator = generator(validation_samples, batch_size=30)

ch, row, col = 3, 80, 320  # Trimmed image format


#NVDIA Model
model = Sequential()
# Preprocess incoming data, 
# Cropping Image to remove sky and car hood
model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(160,320,3))) 
# Normalizing data
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
# 3 Convolution layers with 5x5 kernel size
model.add(Conv2D(24,(5,5),strides = 2,activation='relu'))
model.add(Conv2D(36,(5,5),strides =2, activation='relu'))
model.add(Conv2D(48,(5,5),strides =2,activation='relu'))
# 2 convolutional layers with 
#		filter size 3x3 
#		no subsampling
# 		relu activation
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
# Flatten
model.add(Flatten())
# Four fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose = 1)
model.save('model.h5')
