import tensorflow as tf
import numpy as np
import sys
from tensorflow import keras
y = np.random.randint(0,5,10)
x = tf.random.normal((5,),0,0)
labels = tf.Variable(y)

b = tf.concat([labels, labels], axis=0)
c = tf.concat([[labels, labels]], axis=1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    a = tf.print("labels: ", labels)
    sess.run(a)  # a is a operation
    d = b.eval()
    e = c.eval()
    print("concatenation on axis = 0 -->",d)
    print("concatenation on axis = 1 -->\n", e)