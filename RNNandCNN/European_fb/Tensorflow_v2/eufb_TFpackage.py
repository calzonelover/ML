import sqlite3 as lite
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


# ===================
#  Data preparation
# ===================
# data setting
#dat_dir = '/home/jab/Freetimeproject/Problems/Eufb_2008_2015/'
dat_dir = '/home/u5934/Problem/European_football_2008-2016/'
#dat_dir = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/European_football_2008-2016/'
dat_f_name = 'database.sqlite'
f_dat = dat_dir+dat_f_name
f_score_name = dat_dir+'score_dat.npz' ###
f_wld_name = dat_dir+'wld_dat.npz' ###
n_all = 25976
n_train = 20000
n_test = n_all - n_train # total matches = 25976
biased_value = 0. # fill defect elements
date_split_n = 5 #  has hr/weekday/date/month/year

factor_quantity_name = ['B365H', 'B365D', 'B365A',\
     'BWH', 'BWD', 'BWA', 'WHH', 'WHD', 'WHA']

factor_quality_name = ['league_id', 'season', 'stage', 'home_team_api_id',\
     'away_team_api_id', 'home_player_1', 'home_player_2', 'home_player_3', 'home_player_4',\
      'home_player_5', 'home_player_6', 'home_player_7', 'home_player_8', 'home_player_9',\
       'home_player_10', 'home_player_11', 'away_player_1', 'away_player_2', 'away_player_3',\
        'away_player_4', 'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8',\
         'away_player_9', 'away_player_10', 'away_player_11']

factor_result = ['home_team_goal', 'away_team_goal']

def get_xy_dat_score(f_dat):
    # set
    position_season = date_split_n + len(factor_quantity_name) + 1
    # get file
    con = lite.connect(f_dat)
    with lite.connect(f_dat):
        matches = pd.read_sql_query('SELECT * from Match', con)
    # split intp x,y dat
    x_time = np.array(matches['date'])
    x_quan = np.array(matches[factor_quantity_name[0:len(factor_quantity_name)]])
    x_qual = np.array(matches[factor_quality_name[0:len(factor_quality_name)]])
    # define y value
    y = np.array(matches[factor_result[0:len(factor_result)]])
    # define dummy and put value after
    x_dummy_quan = np.random.rand(y.shape[0], date_split_n + x_quan.shape[1])
    x_dummy_qual = np.random.randint(5, size=(y.shape[0], x_qual.shape[1])) # 5 just unusable random max number
    for i in range(y.shape[0]):
        # change seasen to discreate number
        x_qual[i][1] = season_to_number(x_qual[i][1])
        # insert date
        date_split_i = date_split(x_time[i])
        for j in range(date_split_n):
            x_dummy_quan[i][j] = date_split_i[j]
        # insert quantity value
        for j in range(date_split_n,date_split_n+ len(factor_quantity_name)):
            x_dummy_quan[i][j] = x_quan[i][j - date_split_n ]
            # fix defect column
            if math.isnan(x_quan[i][j - date_split_n ]) or (x_quan[i][j - date_split_n ] != x_quan[i][j - date_split_n ]):
                x_dummy_quan[i][j] = biased_value
        # insert quality value
        for j in range(len(factor_quality_name)):
            if math.isnan(x_qual[i][j]) or (x_qual[i][j]!=x_qual[i][j]):
                x_dummy_qual[i][j] = biased_value
            else:
                x_dummy_qual[i][j] = x_qual[i][j]
    x_dummy_quan.astype(float)
    x_dummy_qual.astype(int)
    y.astype(float)
    x_quan_train, x_qual_train, y_train = x_dummy_quan[:n_train],x_dummy_qual[:n_train],y[0:n_train]
    x_quan_test, x_qual_test, y_test = x_dummy_quan[n_train:matches.shape[0]],\
                                      x_dummy_qual[n_train:matches.shape[0]], y[n_train:matches.shape[0]]
    return x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test 
def get_xy_dat_wld(f_dat):
    # set
    position_season = date_split_n + len(factor_quantity_name) + 1
    # get file
    con = lite.connect(f_dat)
    with lite.connect(f_dat):
        matches = pd.read_sql_query('SELECT * from Match', con)
    # split intp x,y dat
    x_time = np.array(matches['date'])
    x_quan = np.array(matches[factor_quantity_name[0:len(factor_quantity_name)]])
    x_qual = np.array(matches[factor_quality_name[0:len(factor_quality_name)]])
    # define y value
    y = np.array(matches[factor_result[0:len(factor_result)]])
    y_wld = np.random.rand(len(y),3) # defined
    # define dummy and put value after
    x_dummy_quan = np.random.rand(y.shape[0], date_split_n + x_quan.shape[1])
    x_dummy_qual = np.random.randint(5, size=(y.shape[0], x_qual.shape[1])) # 5 just unusable random max number
    for i in range(y.shape[0]):
        # change seasen to discreate number
        x_qual[i][1] = season_to_number(x_qual[i][1])
        # insert date
        date_split_i = date_split(x_time[i])
        for j in range(date_split_n):
            x_dummy_quan[i][j] = date_split_i[j]
        # insert quantity value
        for j in range(date_split_n,date_split_n+ len(factor_quantity_name)):
            x_dummy_quan[i][j] = x_quan[i][j - date_split_n ]
            # fix defect column
            if math.isnan(x_quan[i][j - date_split_n ]) or (x_quan[i][j - date_split_n ] != x_quan[i][j - date_split_n ]):
                x_dummy_quan[i][j] = biased_value
        # insert quality value
        for j in range(len(factor_quality_name)):
            if math.isnan(x_qual[i][j]) or (x_qual[i][j]!=x_qual[i][j]) or x_qual[i][j] < 0:
                x_dummy_qual[i][j] = biased_value
            else:
                x_dummy_qual[i][j] = x_qual[i][j]
        # manage y
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
    x_dummy_quan.astype(float)
    x_dummy_qual.astype(int)
    y.astype(float)
    x_quan_train, x_qual_train, y_train = x_dummy_quan[:n_train],x_dummy_qual[:n_train],y_wld[0:n_train]
    x_quan_test, x_qual_test, y_test = x_dummy_quan[n_train:matches.shape[0]],\
                                       x_dummy_qual[n_train:matches.shape[0]], y_wld[n_train:matches.shape[0]]
    return x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test 
def season_to_number(year_season): # Ex (2008/2009 -> 8)
    return float(year_season[3])
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
def fast_get_xy_dat_score():
    file = np.load(f_score_name)
    x_quan, x_qual, y_score = file['x_quan'], file['x_qual'], file['y_score']
    x_quan_train, x_qual_train, y_train = x_quan[:n_train],x_qual[:n_train],y_score[:n_train]
    x_quan_test, x_qual_test, y_test = x_quan[n_train:n_all],\
                                       x_qual[n_train:n_all], y_score[n_train:n_all]
    return x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test
def fast_get_xy_dat_wld():
    file = np.load(f_wld_name)
    x_quan, x_qual, y_wld = file['x_quan'], file['x_qual'], file['y_wld']
    x_quan_train, x_qual_train, y_train = x_quan[:n_train],x_qual[:n_train],y_wld[:n_train]
    x_quan_test, x_qual_test, y_test = x_quan[n_train:n_all],\
                                       x_qual[n_train:n_all], y_wld[n_train:n_all]
    return x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test

# ===============
#     Model
# ===============

from model_v2 import *



# ==================
#  Training Process
# ==================
learning_rate = 0.0001

def train_nn_model_wld(x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test):
    prediction = model_RNN_v0(input_quan, input_qual) ### need model config
    loss =  tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #learning_rate = 0.001
    # setting
    batch_size = 200
    epochs = 15000
    num_epochs_print = 10
    # file log train
    f_name_acc_rem = os.getcwd()+'/log'+'/logs_train_RNN.olo' ### need model configure
    f_acc_rem = open(f_name_acc_rem, 'a')
    # declare saver
    saver = tf.train.Saver()
    path_saver = os.getcwd()+'/model_repo/model_RNN_v0.ckpt' ### need model configure
    # deploy!!
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # check if model already exist or nit
        if os.path.isfile(path_saver+'.meta'):
            saver.restore(sess, path_saver)
            print('!!!!   model already exist !!!!')
        # start first epoch
        for epoch in range(epochs):
            epoch_loss = 0.
            n_mini_batch = int(len(x_qual_train)/batch_size)
            for mini_batch in range(n_mini_batch):
                epoch_x_quan, epoch_x_qual , epoch_y = \
                    x_quan_train[mini_batch*n_mini_batch:mini_batch*n_mini_batch+n_mini_batch],\
                    x_qual_train[mini_batch*n_mini_batch:mini_batch*n_mini_batch+n_mini_batch],\
                    y_train[mini_batch*n_mini_batch:mini_batch*n_mini_batch+n_mini_batch]
                _, c = sess.run([optimizer, cost], \
                       feed_dict={input_quan: epoch_x_quan, input_qual: epoch_x_qual, y: epoch_y})
                epoch_loss += c # c is in mini_batch
            if num_epochs_print <= epoch:
                print('Epoch',epoch+1,'/',epochs,'loss:',epoch_loss)
            if epochs % epochs/num_epochs_print == 0:
                print('Epoch',epoch+1,'/',epochs,'loss:',epoch_loss)
            f_acc_rem.write('{} {}\n'.format(epoch, epoch_loss))
        # evaluation process
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy on train_set:',accuracy.eval({input_quan: x_quan_train, input_qual: x_qual_train, y: y_train}))
        print('Accuracy on test_set:',accuracy.eval({input_quan: x_quan_test, input_qual: x_qual_test, y: y_test}))
        # save variable (Model)
        saver.save(sess, path_saver)
    # close file log train
    f_acc_rem.close()
        #print(sess.run(prediction, feed_dict={x: x_test[20:25]}), y_test[20:25])
def train_nn_model_score(x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test):
    prediction = model_trivial(x)#model_trivial(x)#model_wld_v1(x)
    cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(prediction, y)))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # setting
    batch_size = 200
    epochs = 200
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




