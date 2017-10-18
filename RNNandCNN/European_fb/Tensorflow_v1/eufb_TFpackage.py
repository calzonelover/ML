import sqlite3 as lite
import pandas as pd
import numpy as np
from datetime import datetime
import time

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell



# ===================
#  Data preparation
# ===================
# data setting
dat_dir = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/European_football_2008-2016/'
dat_f_name = 'database.sqlite'
f_dat = dat_dir+dat_f_name
n_train = 20000
n_test = 26976 - n_train # total matches = 26976
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
    date = ['date']
    date.insert(factor_result)
    print date
    exit()
    x = np.array(matches[date[0:1]+factor_quantity_name[0:len(factor_quantity_name)]\
            +factor_quality_name[0:len(factor_quality_name)]])

    x_quan = np.array(matches[factor_quantity_name[0:len(factor_quantity_name)]])
    x_qual = np.array(matches[factor_quality_name[0:len(factor_quality_name)]])
    # define y value
    y = np.array(matches[factor_result[0:len(factor_result)]])
    # define dummy and put value after
    x_dummy = np.random.rand(y.shape[0], date_split_n + x_quan.shape[0] + x_qual.shape[0])
    for i in range(x_dummy.shape[0]):
        # change seasen to discreate number
        x_qual[i][1] = season_to_number(x_qual[i][1])
        # insert date
        date_split_i = date_split(x_time[i])
        for j in range(date_split_n):
            x[i][j] = date_split_i[j]
        # insert quantity value
        for j in range(date_split_n,date_split_n+ len(factor_quantity_name)):
            x[i][j] = x_quan[i][j - date_split_n ]
        # insert quality value
        for j in range(date_split_n + len(factor_quantity_name), date_split_n + len(factor_quantity_name) + len(factor_quality_name)):
            x[i][j] = x_qual[i][j - (date_split_n + factor_quantity_name.shape[0])]
        # fix defect column
        for j in range(x_dummy.shape[1]):
            if (type(x[i][j]) != int and type(x[i][j]) != float) or x[i][j] != x[i][j]:
                x[i][j] = biased_value
    return x_dummy, y

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

print 'start'
x, y = get_xy_dat_score(f_dat)
print 'end'
print x[19234]









