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
from keras.utils import plot_model
from keras.layers import Input, Dense
from keras.models import Model
import theano

#### Data preparation ###
# data setting
dat_dir = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/European_football_2008-2016/'
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
    # change anyting to float
    x_dummy.astype(float)
    y.astype(float)
    x_train, y_train = x_dummy[:n_train],y[0:n_train]
    x_test, y_test = x_dummy[20000:matches.shape[0]], y[20000:matches.shape[0]]
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
    x_dummy.astype(float)
    y_wld.astype(float)
    x_train, y_train = x_dummy[:n_train],y_wld[0:n_train]
    x_test, y_test = x_dummy[20000:matches.shape[0]], y_wld[20000:matches.shape[0]]
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

#### define model ###
### model with tensorflow
## setting
def RNN(x, weights, biases):
    # settign rnn model
    n_hidden = 20
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)
    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

### simple LSTM with keras
# model setting
batch_size = 32
lr = 0.00025
embd_vec_length = 32
nb_epoch = 1
max_input = 1000
f_model = 'EUsoccer_V0.h5'
verbose = 0
### model with output is win lose and draw
def build_model_wld_v1(input_length, output_length):
    model = Sequential()
    model.add(Embedding(10, 10, input_length = input_length))#input_shape = (None, input_length)))
    model.add(LSTM(100))
    model.add(Dense(output_length, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=lr), metrics = ['accuracy'])
    return model
def build_model_wld(input_length, output_length):
    model = Sequential()
    model.add(LSTM(100, input_shape = (None, input_length)))
    model.add(Dense(output_length, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=lr), metrics = ['accuracy'])
    return model
### model with output is score
def build_model_score(input_length, output_length):
    model = Sequential()
    #model.add(Embedding(max_input ,embd_vec_length, input_length = input_length))
    model.add(LSTM(100, input_shape = (None ,input_length)))
    model.add(Dense(output_length, activation = 'relu'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
    return model
'''
#class model_eu_soccer:
    def __init__(self, batch_size, lr, embd_vec_length, input_length, verbose):
        self.batch_size = batch_size
        self.learning_rate = lr
        self.embd_vec_length = embd_vec_length
        self.max_input = max_input
        self.input_length = input_length
        self.verbose = verbose
        self.model = self.build_model()
    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.max_input, self.embd_vec_length\
            , input_length = self.input_length))
        model.add(LSTM(100))
        model.add(Dense(2, activation = 'relu'))
        model.compile(loss = 'binary_crossentropy'\
            , optimizer = Adam(self.learning_rate), metrices = ['accuracy'])
        return model
    def summary(self):
        print self.model.summary()
    def model_fit(self, x_train, y_train, nb_epoch):
        self.model.fit(x_train, y_train, nb_epoch = nb_epoch, batch_size = self.batch_size)
    def model_save(self, f_model):
        self.model.save(f_model)
    def model_load(self, f_model):
        return self.model.save(f_model)
    def model_evaluate(self, x_test, y_test):
        self.model.evaluate(x_test, y_test, verbose = self.verbose)
'''
