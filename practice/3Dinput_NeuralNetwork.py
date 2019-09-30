import tensorflow as tf
import numpy as np
#tf.enable_eager_execution()
def getTensor(arg):
    return tf.convert_to_tensor(arg, dtype= tf.float32)

m = 3  # Batch Size
tx  = 5 # number of time steps
n_a = 32 # number of RNN hidden units 1st layer
n_s = 32 # number of RNN hidden units 2nd layer
s_prev =  np.random.randn(m, n_s)
a = np.random.randn(m,tx,n_a)


densor1 = tf.layers.Dense(10) # number of hidden units in attention network m, tx, n_units
densor2 = tf.layers.Dense(1) # We need alpha value per time step   m, tx, 1 = alpha for each tx

#Steps
#1. Adjust s_prev shape to make it compatible with a
#2. Perform Dense1 operations on a and s_prev
#3. Compute tanh activations
#4. Perform Dense2 operations  --> m, tx, 1
#5. Perform softmax activation tf.nn.softmax on 2nd axis. default is last axis, i.e. -1. tf.nn.softmax is tensor ops and not layer op
# hence batch dim to be considered while specifying axis (0,1,2) i.e. total 3 axis

s_prev_expanded = tf.expand_dims(s_prev,1)  # m, 1, n_s

score = tf.nn.tanh(densor1(s_prev_expanded) + densor1(a))
print(score.shape)  # (3,5,10)
logits = densor2(score)
print(logits.shape) #(3,5,1)
attention = tf.nn.softmax(logits, axis=1)
print(attention.shape) #(3,5,1)

#print(attention.numpy()) # only with eager execution

print(attention) #Tensor("transpose_3:0", shape=(3, 5, 1), dtype=float64)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess:
    print(attention.eval())


# [[[0.23177488]
#   [0.2889969 ]
#   [0.2152841 ]
#   [0.13676737]
#   [0.12717675]]
#  [[0.15869956]
#   [0.25472575]
#   [0.23835601]
#   [0.09863582]
#   [0.24958286]]
#  [[0.46140363]
#   [0.06772146]
#   [0.13476279]
#   [0.06785225]
#   [0.26825987]]]