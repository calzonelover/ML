import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten
from keras.layers import LSTM,convolutional,MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils

import cPickle
import matplotlib.pyplot as plt

# setting
dir_data = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/cifar-10-batches-py/'
N_output_type = 10
name_saved_model = 'CNN_cifar10.h5'

def order_data(f_name):
    dat=cPickle.load(open(dir_data+f_name,'r'))
    x_test=np.reshape(dat['data'],[dat['data'].shape[0],32,32,3])
    y_test=np.reshape(dat['labels'],[dat['data'].shape[0],1])
    x_test,y_test = x_test.astype('float32'),y_test.astype('float32')
    x_test = np.divide(x_test, 255.)
    y_test = np_utils.to_categorical(y_test, N_output_type)
    return x_test,y_test
def model(x_train):
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(10000,32,32,3)))
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
### main program ###
# prpare data
x_train,y_train = order_data('data_batch_1')
x_test,y_test = order_data('test_batch')
# initialize the model
print x_train.shape#.shape,x_train.shape[0],x_train.shape[1],x_train.shape[2]
#exit()
model=model(x_train)
# train model
model.fit(x_train,y_train, nb_epoch = 1, batch_size = 128, verbose = 1)
model.save(name_saved_model)
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.3f%%" %(score[1]*100))
