import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()
tf.add([[1,2]], [[3,4]])
tf.add([1,2], [3,4])

import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

x = tf.random.uniform((3,3))
print(x)
y = tf.random.uniform([3,3])
print(y)

x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

### Datasets

dat = tf.data.Dataset.from_tensor_slices([1,2,3])
for i in dat:
    print(i)
print("MAP ")
dat = dat.map( lambda x: x+1)
for i in dat:
    prin    t(i)