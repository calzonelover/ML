import tensorflow as tf

a = tf.random_uniform([3,5])

sess = tf.Session()

out = sess.run(a)
print('############ Start ###############')
print(out)