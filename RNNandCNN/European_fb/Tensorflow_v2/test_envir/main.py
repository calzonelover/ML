import tensorflow as tf

from sub import *


sess = tf.Session()
sess.run(tf.global_variables_initializer())

bla = sess.run(x, feed_dict={x: [1,2,3]})
print bla