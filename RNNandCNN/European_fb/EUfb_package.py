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

#### function to clean data ###
# data setting 
dat_dir = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/European_football_2008-2016/'
dat_f_name = 'database.sqlite'
f_dat = dat_dir+dat_f_name
f_match_factors = 'match_factors.olo'
biased_value = -1000. # biased value when elements in array does not contain int or float
n_train = 20000
n_test =  25976 - n_train # total matches - 26976
# define function
def get_xy_dat(f_dat, f_match_factors):
    # get file
    con = lite.connect(f_dat)
    # fix last \n array point
    match_str = open(f_match_factors).read().split(', ')
    dummy = match_str[len(match_str)-1]
    length_last_array = len(match_str[len(match_str)-1])-1
    match_str[len(match_str)-1] = match_str[len(match_str)-1][0:length_last_array]
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
    for i in range(len(x)):
        x[i][3] = season_to_number(x[i][3])
        x[i][5] = date_to_sTime(x[i][5])-date_to_sTime(x[0][5])
        # then elimintate useless data which not int or float
        for j in range(len(x[i])):
        	if (type(x[i][j]) != int and type(x[i][j]) != float) or x[i][j] != x[i][j]:
        		x[i][j] = biased_value
    # change anyting to float
    x.astype(float)
    y.astype(float)
    x_train, y_train = x[:n_train],y[0:n_train]
    x_test, y_test = x[20000:matches.shape[0]], y[20000:matches.shape[0]]
    return x_train, y_train, x_test, y_test
def season_to_number(year_season): # with pattern '2008/2009' (2008/2009 -> 8.5)
    return 0.5*(float(year_season[3])+float(year_season[8]))
def date_to_sTime(date_want):# with pattern '2008-08-17 00:00:00' to second
    date_want = datetime.strptime(str(date_want), "%Y-%m-%d %H:%M:%S")
    return float(time.mktime(date_want.timetuple()))

#### define model ###
# model setting
batch_size = 32
lr = 0.00025
embd_vec_length = 32
nb_epoch = 1
max_input = 1000
f_model = 'EUsoccer_V0.h5'
verbose = 0
def build_model(input_length, max_input, embd_vec_length, output_length):
    model = Sequential()
    model.add(Embedding(max_input ,embd_vec_length, input_length = input_length))
    model.add(LSTM(100))
    model.add(Dense(output_length, activation = 'relu'))
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr))
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















