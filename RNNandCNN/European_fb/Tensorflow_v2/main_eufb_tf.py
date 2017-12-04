#!/usr/bin/env python
from eufb_TFpackage import *


### get data ###
x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = fast_get_xy_dat_wld()
#x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = fast_get_xy_dat_score()
#x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = get_xy_dat_wld(f_dat)
#x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = get_xy_dat_score(f_dat)

#input_length = x_train.shape[1]
#output_length = y_train.shape[1]


print(x_quan_train.shape,x_qual_train.shape,y_train.shape)
print(x_quan_test.shape,x_qual_test.shape,y_test.shape)


train_RNN_model_wld(x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test)
#train_CNN_RNN_model_wld(x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test)
#train_multilayer_model_wld(x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test)



##### train score #############
#x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = fast_get_xy_dat_score()
#train_nn_model_score(x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test)











