# data setting 
dat_dir = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/European_football_2008-2016/'
dat_f_name = 'database.sqlite'
f_dat = dat_dir+dat_f_name
f_match_factors = 'match_factors.olo'
biased_value = -1000. # biased value when elements in array does not contain int or float
n_train = 20000
n_test =  25976 - n_train # total matches - 26976
# model setting
batch_size = 64
lr = 0.00025
embd_vec_length = 64
nb_epoch = 1
max_input = 1000

from EUfb_package import *

###### start ##############
# take the eu soccer data
x_train, y_train, x_test, y_test = get_xy_dat(f_dat, f_match_factors)
# thumb
input_length = x_train.shape[1]
output_length = y_train.shape[1]
model = build_model(input_length, max_input, embd_vec_length, output_length)
print model.summary()
model.fit(x_train, y_train, epochs = nb_epoch, batch_size = batch_size)
# save model
model.save(f_model)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))