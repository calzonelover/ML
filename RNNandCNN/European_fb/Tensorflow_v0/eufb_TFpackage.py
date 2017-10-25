import sqlite3 as lite
import pandas as pd
import numpy as np
from datetime import datetime
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
#from keras.utils import plot_model
from keras.layers import Input, Dense
from keras.models import Model

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# ==================
# Data preparation 
# ==================
# data setting
#dat_dir = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/European_football_2008-2016/'
dat_dir = '/root/Problems/Eufb_2008_2016/'
dat_f_name = 'database.sqlite'
f_dat = dat_dir+dat_f_name
f_match_factors = 'match_factors.olo'
biased_value = 0. # biased value when elements in array does not contain int or float
n_train = 20000
n_test =  26976 - n_train # total matches - 26976
# data preparation
def get_xy_dat_score(f_dat, f_match_factors):
    # get file
    con = lite.connect(f_dat)
    # fix last \n array point
    match_str = open(f_match_factors).read().split(', ')
    dummy = match_str[len(match_str)-1]
    match_str[len(match_str)-1] = match_str[len(match_str)-1][0:3]
    # get data
    with lite.connect(f_dat):
        matches = pd.read_sql_query('SELECT * from Match', con)
        # optional
        #countries = pd.read_sql_query('SELECT * from Country', con)
        #leagues = pd.read_sql_query('SELECT * from League', con)
        #teams = pd.read_sql_query('SELECT * from Team', con)
        #teams_att = pd.read_sql_query('SELECT * from Team_Attributes', con)
        #players = pd.read_sql_query('SELECT * from Player', con)
        #players_att = pd.read_sql_query('SELECT * from Player_Attributes', con)
    # split into x,y data
    x, y = np.array(matches[match_str[0:9]+match_str[11:len(match_str)+1]])\
        , np.array(matches[match_str[9:11]]) # array 9 to 11
    x_dummy = np.random.rand(x.shape[0],x.shape[1]+4) # date should have 5 component ## fix_date
    x_dummy2 = np.random.rand(x.shape[0],x.shape[1]+4-9) ###
    # fix date and season to just number
    for i in range(len(x)):
        x[i][3] = season_to_number(x[i][3])
        insert_date = date_split(x[i][5])
        x[i][5] = float(x[i][5][0:4])
        # then elimintate useless data which not int or float
        for j in range(len(x[i])):
        	if (type(x[i][j]) != int and type(x[i][j]) != float) or x[i][j] != x[i][j]:
        		x[i][j] = biased_value
        ### redefine x_dummy
        for k in range(0,4+1):
            x_dummy[i][k] = x[i][k]
        for k in range(5,9+1):
            x_dummy[i][k] = insert_date[k-5]
        for k in range(9,len(x_dummy[i])):
            x_dummy[i][k] = x[i][k-4]
    x_dummy2[i] = np.delete(x_dummy[i], [97,98,99,103,104,105,106,107,108]) ###
    # change anyting to float
    x_dummy2.astype(float) ###
    y.astype(float)
    x_train, y_train = x_dummy2[:n_train],y[0:n_train]
    x_test, y_test = x_dummy2[20000:matches.shape[0]], y[20000:matches.shape[0]]
    return x_train, y_train, x_test, y_test

def get_xy_dat_wld(f_dat, f_match_factors):
    # get file
    con = lite.connect(f_dat)
    # fix last \n array point
    match_str = open(f_match_factors).read().split(', ')
    dummy = match_str[len(match_str)-1]
    match_str[len(match_str)-1] = match_str[len(match_str)-1][0:3]
    # get data
    with lite.connect(f_dat):
        matches = pd.read_sql_query('SELECT * from Match', con)
        # optional
        #countries = pd.read_sql_query('SELECT * from Country', con)
        #leagues = pd.read_sql_query('SELECT * from League', con)
        #teams = pd.read_sql_query('SELECT * from Team', con)
        #teams_att = pd.read_sql_query('SELECT * from Team_Attributes', con)
        #players = pd.read_sql_query('SELECT * from Player', con)
        #players_att = pd.read_sql_query('SELECT * from Player_Attributes', con)
    # split into x,y data
    x, y = np.array(matches[match_str[0:9]+match_str[11:len(match_str)+1]])\
        , np.array(matches[match_str[9:11]]) # array 9 to 11
    # fix date and season to just number
    x_dummy = np.random.rand(x.shape[0],x.shape[1]+4) # date should have 5 component ## fix_date
    x_dummy2 = np.random.rand(x.shape[0],x.shape[1]+4-9) ###
    y_wld = np.random.rand(len(y),3) # defined
    for i in range(len(x)):
        x[i][3] = season_to_number(x[i][3])
        insert_date = date_split(x[i][5])
        x[i][5] = float(x[i][5][0:4])
        # then elimintate useless data which not int or float
        for j in range(len(x[i])):
        	if (type(x[i][j]) != int and type(x[i][j]) != float) or x[i][j] != x[i][j]:
        		x[i][j] = biased_value
        ### redefine x_dummy
        for k in range(0,4+1):
            x_dummy[i][k] = x[i][k]
        for k in range(5,9+1):
            x_dummy[i][k] = insert_date[k-5]
        for k in range(9,len(x_dummy[i])):
            x_dummy[i][k] = x[i][k-4]
        x_dummy2[i] = np.delete(x_dummy[i], [97,98,99,103,104,105,106,107,108])###
        # make score to result win lose and draw
        if y[i][0] > y[i][1]:
            y_wld[i][0] = 1.
            y_wld[i][1] = 0.
            y_wld[i][2] = 0.
        if y[i][0] < y[i][1]:
            y_wld[i][0] = 0.
            y_wld[i][1] = 1.
            y_wld[i][2] = 0.
        if y[i][0] == y[i][1]:
            y_wld[i][0] = 0.
            y_wld[i][1] = 0.
            y_wld[i][2] = 1.
    # change anyting to float
    x_dummy2.astype(float)
    y_wld.astype(float)
    x_train, y_train = x_dummy2[:n_train],y_wld[0:n_train] ###
    x_test, y_test = x_dummy2[20000:matches.shape[0]], y_wld[20000:matches.shape[0]]###
    return x_train, y_train, x_test, y_test
def season_to_number(year_season): # with pattern '2008/2009' (2008/2009 -> 8.5)
    return 0.5*(float(year_season[3])+float(year_season[8]))
def date_split(date_want):# with pattern '2008-08-17 00:00:00' to hr/weekday/date/month/year
    year_want = float(date_want[0:4])
    month_want = float(date_want[5:7])
    dates_want = float(date_want[8:10])
    hr_want = float(date_want[11:13])
    min_want = float(date_want[14:16])
    sec_want = float(date_want[17:19])
    weekday_want = datetime(int(year_want), int(month_want), int(dates_want),\
        int(hr_want),int(min_want),int(sec_want)).weekday()
    return np.array([hr_want, weekday_want, dates_want, month_want, year_want])



# ========================
# model for Win Lose Draw
# ========================

n_feature = 100
n_classes = 3 # 2 = scores, 3 = Win/lose/equal
x = tf.placeholder('float', [None, n_feature])
y = tf.placeholder('float', [None, n_classes])
# this model layer_left*x*layer_right -> LSTM

def model_RNN_v0(data):
    n_input = 100
    n_hidden = 100
    n_classes = 3
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    x = tf.split(x,n_input,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden),rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.softmax(output)
    return output

def model_RNN_v1(data):
    n_input = 100
    n_hidden = 100
    n_classes = 3
    hd_layer1 = {'weights':tf.Variable(tf.random_normal([n_input,n_input]))}
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                  'biases': tf.Variable(tf.random_normal([n_classes]))}
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, n_input])
    new_x = tf.matmul(x, hd_layer1['weights']) # dictionary like, got the same shape
    # Generate a n_input-element sequence of inputs
    new_x = tf.split(new_x,n_input,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden),rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, new_x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.softmax(output)
    return output

def model_CNN_v0(data):
    n_input = 100
    n_hidden = 100
    n_classes = 3
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, 10, n_input/10, 1])
    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv1) # shape [None, ...]
    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate = 0.4)
    # Output layer, class prediction
    output = tf.layers.dense(fc1, n_classes)
    output = tf.nn.softmax(output)
    return output
def model_CNN_v1(data):
    n_input = 100
    n_hidden = 100
    n_classes = 3
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, 10, n_input/10, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=x,
                            filters=32,
                            kernel_size=[3, 3],
                            padding="same",
                            activation=tf.nn.sigmoid)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.contrib.layers.flatten(pool2)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    # Output layer, class prediction
    output = tf.layers.dense(logits, n_classes)
    output = tf.nn.softmax(output)
    return output

def model_CNN_RNN_v0(data):
    n_input = 100
    n_hidden = 100
    n_hidden_2 = 100
    n_classes = 3
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))} ####
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, 10, n_input/10, 1])
    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv1) # shape [None, ...]
    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate = 0.4)
    # Output layer, class prediction
    x = tf.layers.dense(fc1, n_hidden_2)
    x = tf.reshape(data, [-1, n_hidden_2])
    # Generate a n_input-element sequence of inputs
    new_x = tf.split(x,n_hidden_2,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden),rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, new_x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.softmax(output)
    return output

def model_CNN_RNN_v1(data):
    n_input = 100
    n_hidden = 100
    n_hidden_2 = 100
    n_classes = 3
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))} ####
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, 10, n_input/10, 1]) # want to reshape 100 -> 10x10x1
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=x,
                            filters=32,
                            kernel_size=[3, 3],
                            padding="same",
                            activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.contrib.layers.flatten(pool2)

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    # Logits Layer
    x = tf.layers.dense(dropout, n_hidden_2)
    x = tf.reshape(data, [-1, n_hidden_2])
    # Generate a n_input-element sequence of inputs
    new_x = tf.split(x,n_hidden_2,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden),rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, new_x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.softmax(output)
    return output
'''
def model_RNN_CNN(data):
    n_input = 109
    n_hidden = 100
    n_classes = 3
    hd_layer_1 = {}
'''
def model_trivial(data):
    n_nodes_hl1 = 128
    n_nodes_hl2 = 128
    n_nodes_hl3 = 128
    n_feature = 100
    n_classes = 3
    hd_layer1 = {'weights':tf.Variable(tf.random_normal([n_feature,n_nodes_hl1]))
                 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hd_layer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]))
                 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hd_layer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]))
                 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes]))
                    }#,'biases':tf.Variable(tf.random_normal([n_classes]))}
    # ( A*x + b )
    l1 = tf.add(tf.matmul(data, hd_layer1['weights']), hd_layer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hd_layer2['weights']), hd_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hd_layer3['weights']), hd_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.nn.softmax(tf.matmul(l3, output_layer['weights']))#, output_layer['biases'])
    return output




# =================
# model for scroes
# =================

def model_trivial_score(data):
    n_nodes_hl1 = 128
    n_nodes_hl2 = 128
    n_nodes_hl3 = 128
    n_feature = 100
    n_classes = 2
    hd_layer1 = {'weights':tf.Variable(tf.random_normal([n_feature,n_nodes_hl1]))
                 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hd_layer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]))
                 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hd_layer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]))
                 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes]))
                    }#,'biases':tf.Variable(tf.random_normal([n_classes]))}
    # ( A*x + b )
    l1 = tf.add(tf.matmul(data, hd_layer1['weights']), hd_layer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hd_layer2['weights']), hd_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hd_layer3['weights']), hd_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.nn.relu(tf.matmul(l3, output_layer['weights']))#, output_layer['biases'])
    return output
def model_RNN_v0_scores(data):
    n_input = 100
    n_hidden = 100
    n_classes = 2
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                  'biases': tf.Variable(tf.random_normal([n_classes]))}
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    x = tf.split(x,n_input,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden),rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.relu(output)
    return output
def model_RNN_origin_scores(data):
    n_input = 100
    n_hidden = 100
    n_classes = 2
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                  'biases': tf.Variable(tf.random_normal([n_classes]))}
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    x = tf.split(x,n_input,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.relu(output)
    return output
def model_CNN_v0_scores(data):
    n_input = 100
    n_hidden = 100
    n_classes = 2
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, 10, n_input/10, 1])
    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv1) # shape [None, ...]
    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate = 0.4)
    # Output layer, class prediction
    output = tf.layers.dense(fc1, n_classes)
    output = tf.nn.relu(output)
    return output
def model_CNN_RNN_v0_scores(data):
    n_input = 100
    n_hidden = 100
    n_hidden_2 = 100
    n_classes = 2
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))} ####
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, 10, n_input/10, 1])
    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(conv1) # shape [None, ...]
    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate = 0.4)
    # Output layer, class prediction
    x = tf.layers.dense(fc1, n_hidden_2)
    x = tf.reshape(data, [-1, n_hidden_2])
    # Generate a n_input-element sequence of inputs
    new_x = tf.split(x,n_hidden_2,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden),rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, new_x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.relu(output)
    return output

def model_CNN_RNN_v1_scores(data):
    n_input = 100
    n_hidden = 100
    n_hidden_2 = 100
    n_classes = 2
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))} ####
    # reshape to [1, n_input]
    x = tf.reshape(data, [-1, 10, n_input/10, 1]) # want to reshape 100 -> 10x10x1
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=x,
                            filters=32,
                            kernel_size=[3, 3],
                            padding="same",
                            activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.contrib.layers.flatten(pool2)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    # Logits Layer
    x = tf.layers.dense(dropout, n_hidden_2)
    x = tf.reshape(data, [-1, n_hidden_2])
    # Generate a n_input-element sequence of inputs
    new_x = tf.split(x,n_hidden_2,1)
    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cells = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(n_hidden),rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cells, new_x, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.relu(output)
    return output
# ==================
# Training process
# ==================

def train_nn_model_wld(x_train, y_train, x_test, y_test):
    n_feature = x_train.shape[1]
    prediction = model_CNN_RNN_v1(x)#model_trivial(x)#model_wld_v1(x)
    loss =  tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost) #learning_rate = 0.001
    # setting
    batch_size = 200
    epochs = 3
    # deploy!!
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0.
            n_mini_batch = int(len(x_train)/batch_size)
            for mini_batch in range(int(len(x_train)/batch_size)):
                epoch_x, epoch_y = x_train[mini_batch*n_mini_batch:mini_batch*n_mini_batch+n_mini_batch]\
                    , y_train[mini_batch*n_mini_batch:mini_batch*n_mini_batch+n_mini_batch]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c # c is in mini_batch
            print('Epoch',epoch+1,'/',epochs,'loss:',epoch_loss)
        # evaluation process
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))
        print(sess.run(prediction, feed_dict={x: x_test[20:25]}), y_test[20:25])

def train_nn_model_scores(x_train, y_train, x_test, y_test):
    n_feature = x_train.shape[1]
    prediction = model_CNN_RNN_v1_scores(x)#model_trivial(x)#model_wld_v1(x)
    cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, y)))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # setting
    batch_size = 200
    epochs = 3
    # deploy!!
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0.
            n_mini_batch = int(len(x_train)/batch_size)
            for mini_batch in range(int(len(x_train)/batch_size)):
                epoch_x, epoch_y = x_train[mini_batch*n_mini_batch:mini_batch*n_mini_batch+n_mini_batch]\
                    , y_train[mini_batch*n_mini_batch:mini_batch*n_mini_batch+n_mini_batch]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch',epoch+1,'/',epochs,'loss:',epoch_loss)
        # evaluation process
        accuracy = tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, y)))
        print('RMS:',accuracy.eval({x:x_test, y:y_test}))
        print(sess.run(prediction, feed_dict={x: x_test[12:20]}), y_test[12:20])




















