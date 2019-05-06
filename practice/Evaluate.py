import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import f1_score
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = np.array(x_train).reshape(x_train.shape[0],-1)
x_test = np.array(x_test).reshape(x_test.shape[0],-1)
model = tf.keras.Sequential([
keras.layers.Dense(units=128,activation=tf.nn.relu),
keras.layers.Dense(10, activation = tf.nn.softmax)
])

# loss='sparse_categorical_crossentropy'

# loss=tf.losses.sparse_softmax_cross_entropy
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x_train/255,y_train,epochs=3,verbose=1)
# Notice we need to retain the matrix shape in evalute and predict i.e. (1,n) and not n
acc = model.evaluate(x_train[0:1]/255, y_train[0:1])
pred = model.predict(x_train[0:1]/255)
