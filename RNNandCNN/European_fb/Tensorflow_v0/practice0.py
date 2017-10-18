'''
input > weight > hd layer 1 > weight > hd layer 2 > weight > output

compare output to intenedded output > loss function (cross entropy)
optimization fn (optimizer) > minimize cost (AdamOptimizer, SGD, AdaGrad)

backprop.

feed forward + backprob = epoch


'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot = True)

# setting
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def nn_model(data):
	hd_layer1 = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1]))
				 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hd_layer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]))
				 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hd_layer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]))
				 ,'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes]))
					,'biases':tf.Variable(tf.random_normal([n_classes]))}
	# ( A*x + b )
	l1 = tf.add(tf.matmul(data, hd_layer1['weights']), hd_layer1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hd_layer2['weights']), hd_layer2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hd_layer3['weights']), hd_layer3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
	return output

def train_nn_model(x):
	prediction = nn_model(x)
	loss =  tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y)
	cost = tf.reduce_mean(loss)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
	# setting
	batch_size = 100
	epochs = 3
	chunk_size = 28
	n_chunks = 28
	# Let's learn !
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer()) # init any fucking variable
		for epoch in range(epochs):
			epoch_loss = 0.
			for mini_batch in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch',epoch,'/',epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks*chunk_size)), y:mnist.test.labels}))

train_nn_model(x)
