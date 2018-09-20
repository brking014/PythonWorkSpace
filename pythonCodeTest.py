import tensorflow as tf
import numpy as np

# a
# [[1, 2, 3],
#  [4, 5, 6]]
# a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
a = np.float(np.random.rand(2, 100))

# b1
# [[ 7,  8],
# [ 9, 10],
# [11, 12]]
# b1 = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
b1 = np.float32(tf.random_uniform([1, 2], -1.0, 1.0))

# b2
# [[ 7  8  9]
# [10 11 12]]
b2 = tf.constant([7, 8, 9, 10, 11, 12], shape=[2, 3])

# c???? ?????????column????????????row?
# [[ 58,  64],
#  [139, 154]]
c = tf.matmul(a, b1)
with tf.Session():
    print(c.eval())

# d`???????
# [[ 7 16 27]
# [40 55 72]] d = tf.multiply(a, b2) #?????? with tf.Session():
