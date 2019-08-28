import tensorflow as tf
import numpy as np
from tensorflow import keras

#### PREPARE DATASET ####
mnist_dset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dset.load_data()
x_train = x_train[0:10000]
y_train = y_train[0:10000]
K = tf.keras.backend
if K.image_data_format() == 'channels_first':
    K.set_image_data_format('channels_last')
assert K.image_data_format() == 'channels_last'

#### NORMALIZE INPUT ####
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#### ADD CHANNEL dimension which is not present in grayscale ####
x_train = np.expand_dims(x_train,axis=-1)
x_test = np.expand_dims(x_test,axis=-1)

assert x_train[0].shape == (28,28,1)
assert x_test[0].shape == (28,28,1)

#### BUILD MODEL ####

model = tf.keras.Sequential([
keras.layers.Flatten(),
keras.layers.Dense(units=128,activation=tf.keras.activations.relu),
keras.layers.Dense(10, activation=tf.keras.activations.softmax)  #Output layer is in one hot form
])

##### sparse_categorical_crossentropy ####
print("without onehotencoding y_train & y_test")
assert y_train.shape == (10000,)
assert y_test.shape == (10000,)
# loss='sparse_categorical_crossentropy'
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2, verbose=0 )


y_test_0_sparse = model.predict(np.expand_dims(x_test[0], axis=0))
print(np.round(y_test_0_sparse))

#### categorical_crossentropy ####

print("with onehot encoding  y_train & y_test")
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)
assert y_train.shape == (10000,10)
assert y_test.shape == (10000,10)

# loss='sparse_categorical_crossentropy'
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2, verbose=0 )

y_test_0_onehot = model.predict(np.expand_dims(x_test[0], axis=0))
print(np.round(y_test_0_onehot))

#### if target Y is an integer tensor, use sparse_categorical_crossentropy and without onehotencoding