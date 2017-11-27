import numpy as np
import tensorflow as tf

dat_dir = '/home/default/ML/Jab/Problems/European_football_2008-2016/'
f_wld_name = dat_dir+'wld_dat.npz' ###
n_all = 25976
n_train = 24000
n_test = n_all - n_train # total matches = 25976

def fast_get_xy_dat_wld():
    file = np.load(f_wld_name)
    x_quan, x_qual, y_wld = file['x_quan'], file['x_qual'], file['y_wld']
    x_quan_train, x_qual_train, y_train = x_quan[:n_train],x_qual[:n_train],y_wld[:n_train]
    x_quan_test, x_qual_test, y_test = x_quan[n_train:n_all],\
                                       x_qual[n_train:n_all], y_wld[n_train:n_all]
    return x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test
def shuffle_sequence_input(WantShuf_x_quan, WantShuf_x_qual, WantShuf_y):
    # get size of input (Numpy case)
	n_quan = WantShuf_x_quan.shape[1]
	n_qual = WantShuf_x_qual.shape[1]
	n_y = WantShuf_y.shape[1]
	## chaneg x_qual before combine
	#WantShuf_x_qual = tf.to_float(WantShuf_x_qual)
	# combined tensor
	WantShuf = np.concatenate([WantShuf_x_quan, WantShuf_x_qual, WantShuf_y], 1)
	# shuffle sequence
	np.random.shuffle(WantShuf)
	Shuffle_Quan = WantShuf[:,0:n_quan]
	Shuffle_Qual = WantShuf[:,n_quan:n_quan+n_qual]
	Shuffle_y = WantShuf[:,n_quan+n_qual:]
	return Shuffle_Quan, Shuffle_Qual, Shuffle_y
# get dat
x_quan_train, x_qual_train, y_train, x_quan_test, x_qual_test, y_test = fast_get_xy_dat_wld()

print(x_quan_train[20100], x_qual_train[20100], y_train[20100])
#print(x_quan_train.shape, x_qual_train.shape, y_train.shape[1])
#exit()
new_x_quan_train, new_x_qual_train, new_y_train = shuffle_sequence_input(x_quan_train, x_qual_train, y_train)
#sess = tf.Session()
#new_x_quan_train, new_x_qual_train, new_y_train = sess.run([new_x_quan_train, new_x_qual_train, new_y_train])
print(new_x_quan_train.shape, new_x_qual_train.shape, new_y_train.shape)
print(new_x_quan_train[20100], new_x_qual_train[20100], new_y_train[20100])

	#sess_shuffle = tf.Session()
	#Shuffle_Quan, Shuffle_Qual, Shuffle_y = sess_shuffle.run([Shuffle_Quan, Shuffle_Qual, Shuffle_y])