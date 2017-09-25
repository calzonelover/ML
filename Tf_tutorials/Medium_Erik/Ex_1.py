import tensorflow as tf

hello = tf.constant('Hello, tensorflow')

sess = Session()

print sess.run(hello)
