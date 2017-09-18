import numpy as np
from keras.datasets import mnist # data set of number
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten
from keras.layers import LSTM,convolutional,MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils

import matplotlib.pyplot as plt

# setting
batch_size = 128
nb_epoch = 3
name_saved_model = 'CNN_on_MNIST.h5'
N_output_type = 10
# model
def model(x_train):
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(x_train[0].shape[0],x_train[0].shape[1],1)))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    print model.summary()
    return model
### program start ###
print 'program start'
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# let normalize and output to float
x_train = np.divide(x_train,255.)
x_test = np.divide(x_test,255.)
y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
# let 1->10 to binary array 10 elements
y_train = np_utils.to_categorical(y_train, N_output_type)
y_test = np_utils.to_categorical(y_test, N_output_type)
# set 28x28 to 28x28x1
x_train = x_train.reshape(x_train.shape[0],x_train[0].shape[0],x_train[0].shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test[0].shape[0],x_test[0].shape[1],1)
# activate model
model=model(x_train)
# fit model
model.fit(x_train, y_train, nb_epoch = 1, batch_size = 128, verbose=1)
model.save(name_saved_model)
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.3f%%" % (score[1]*100))
