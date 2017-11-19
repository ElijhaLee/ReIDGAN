import tensorflow as tf

x = tf.placeholder(tf.float32, [1, 3])
y = tf.get_variable('w', [3, 2], dtype=tf.float32)

z = tf.cast(tf.argmax(tf.matmul(x, y), 1), tf.float32) + tf.reduce_mean(tf.matmul(x, y))
o = tf.train.GradientDescentOptimizer(0.001)
o.minimize(z, var_list=[y])
