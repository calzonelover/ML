from eufb_TFpackage import *

### get data ###
#x_train, y_train, x_test, y_test = get_xy_dat_score(f_dat)
x_train, y_train, x_test, y_test = get_xy_dat_wld(f_dat)

input_length = x_train.shape[1]
output_length = y_train.shape[1]

print(x_train.shape,y_train.shape)
#exit()
#x = tf.placeholder('float',[None, input_length])
#y = tf.placeholder('float', [None, output_length])
#x_train = np.reshape(x_train, [x_train.shape[0],input_length,1])### for lstm
#x_test = np.reshape(x_test, [x_test.shape[0],input_length,1])### for lstm

train_nn_model_wld(x_train, y_train, x_test, y_test)
#train_nn_model_score(x_train, y_train, x_test, y_test)



















