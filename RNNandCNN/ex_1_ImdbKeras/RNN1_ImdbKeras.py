import numpy as np
from keras.datasets import imdb # review
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
# setting
batch_size = 64
max_review_length = 500 # truncate and pad input sequences (clear 0 and use only 500)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000 # max word only 5000 words in this dictionary
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 4
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
print model.summary()
### test
print y_train.shape
###
model.fit(X_train, y_train, nb_epoch=3, batch_size=batch_size)
# save model
model.save('ImdbReview_PN_1.h5')
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
