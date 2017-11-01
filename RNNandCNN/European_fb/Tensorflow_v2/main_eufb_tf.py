from eufb_TFpackage import *


### get data ###
x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = get_xy_dat_wld(f_dat)
#x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = get_xy_dat_score(f_dat)

#input_length = x_train.shape[1]
#output_length = y_train.shape[1]


print(x_quan_train.shape,x_qual_train.shape,y_train.shape)
print(x_quan_test.shape,x_qual_test.shape,y_test.shape)

#print(x_qual_train[19234])
#print(x_quan_train[19234])
#exit()
#sess = tf.Session()
#print(sess.run(tf.slice(x_qual_train,[0,3],[-1,21]))[19234])
#exit()

train_nn_model_wld(x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test)
#train_nn_model_score(x_train, y_train, x_test, y_test)
















