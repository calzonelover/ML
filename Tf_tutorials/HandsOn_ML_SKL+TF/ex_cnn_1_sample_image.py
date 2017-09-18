import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt
# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape
# Create 2 filters
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 0] = 1 # vertical line
filters_test[3, :, :, 1] = 1 # horizontal line
# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides=[1,2,2,1], padding="SAME")
sess=tf.Session()
output = sess.run(convolution, feed_dict={X: dataset})
plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map plt.show()
plt.show()
