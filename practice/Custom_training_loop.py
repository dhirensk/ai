import tensorflow as tf
import numpy as np
#tf.enable_eager_execution()

from tensorflow.keras.datasets import  mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#Add channels for GrayScale
X_train = np.expand_dims(X_train, axis = -1).astype(np.float32)
X_test = np.expand_dims(X_test, axis = -1).astype(np.float32)
Y_train = np.expand_dims(Y_train,-1)
Y_test = np.expand_dims(Y_test, -1)
class Mnist(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        X = self.conv_1(inputs)
        X = self.conv_2(X)
        X= self.flatten(X)
        X = self.dense1(X)
        return X

model = Mnist()
optimizer = tf.keras.optimizers.Adam()
#model.compile(optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
#model.fit(x=X_train, y=Y_train, validation_data=[X_test, Y_test], batch_size=64, epochs=2)

#Custom training loop
def train_on_batch(x, y):
    return x
