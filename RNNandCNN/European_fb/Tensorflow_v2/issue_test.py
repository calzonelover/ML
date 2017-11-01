import tensorflow as tf

vocab_size = 1000000
embed_size = 3

embed_sp = tf.Variable(tf.random_uniform([vocab_size, embed_size],-1.0,1.0))

input = tf.placeholder(tf.int32, shape=[4])

embed = tf.nn.embedding_lookup(embed_sp, input)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

bla = sess.run([embed], feed_dict={input:[1,2,3,4]})
print(bla)
