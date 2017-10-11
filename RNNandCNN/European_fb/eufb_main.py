# data setting
dat_dir = '/Users/Macintosth/Desktop/FreeTimeProject/Problem/European_football_2008-2016/'
dat_f_name = 'database.sqlite'
f_dat = dat_dir+dat_f_name
f_match_factors = 'match_factors.olo'
biased_value = -1000. # biased value when elements in array does not contain int or float
n_train = 20000
n_test =  26976 - n_train # total matches - 26976

from EUfb_package import *

###### start ##############
# take the eu soccer data
#x_train, y_train, x_test, y_test = get_xy_dat_score(f_dat, f_match_factors)
x_train, y_train, x_test, y_test = get_xy_dat_wld(f_dat, f_match_factors)
# thumb
input_length = x_train.shape[1]
output_length = y_train.shape[1]
x_train = np.reshape(x_train, [x_train.shape[0], 1, x_train.shape[1]])
x_test = np.reshape(x_test, [x_test.shape[0], 1, x_test.shape[1]])
print x_train.shape,y_train.shape

#model = build_model_score(input_length, output_length)
model = build_model_wld(input_length, output_length)
#model = build_model_wld_v1(input_length, output_length)

#plot_model(model, to_file='model.png') ###

print model.summary()
model.fit(x_train, y_train, epochs = 3, batch_size = batch_size, verbose=1, validation_data=(x_test, y_test))
# save model
model.save(f_model)
# Final evaluation of the model
print 'test'
scores = model.evaluate(x_test, y_test, verbose=0)
print scores
#print("Accuracy: %.2f%%" % (scores[1]*100))
