# SparseCategoricalCrosssEntropy
import tensorflow as tf
import numpy as np
# y_true is sparse integer output containing class ids
y_true = np.array([0,1,2,1]) # 3 classes
y_true= tf.convert_to_tensor(y_true, dtype=tf.int32)

#y_pred can be logits i.e. result of Wx+B for each class or it can be class probabilities distribution i.e. after applying a softmax
y_pred = np.array([[1.25, 0.23, 0.04], [0.03,0.81, 0.24], [0.2, 0.34, 1.23], [0.29, 1.55, 0.1]])
y_pred= tf.convert_to_tensor(y_pred, dtype=tf.float32)

def loss_function(y_true, y_pred):
 # not calculating loss when input class is 0
  mask = tf.math.logical_not(tf.math.equal(y_true, 0))
  #loss_ = tf.keras.losses.sparse_categorical_crossentropy( from_logits=True,y_true= y_true, y_pred = y_pred)
  scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  loss = scc(y_true,y_pred)
  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask
  return tf.reduce_mean(loss), loss

total_loss, loss = loss_function(y_true,y_pred)

with tf.Session() as sess:
    sess.run(total_loss)
    print(total_loss.eval())
    sess.run(loss)
    print(loss.eval())
