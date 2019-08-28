import tensorflow as tf
import os
k = tf.placeholder(tf.float32)

random_normal =  tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
tf.summary.histogram("moving_mean_normal",random_normal)
# image Tensor must be 4-D with last dim 1, 3, or 4
img_tensor = tf.summary.image("image tensor",tf.random_normal(shape=[1,56,56,1],mean=0,stddev=1.0))
logdir = os.getcwd()+ "\\histogram_example"


sess = tf.Session()
writer = tf.summary.FileWriter(logdir=logdir)
summary = tf.summary.merge_all()

N =6
for i in range(N):
    val = i/float(N)
    summ = sess.run(summary, feed_dict={k: val})
    writer.add_summary(summ,global_step=i)
img = sess.run(img_tensor)
writer.add_summary(img)
