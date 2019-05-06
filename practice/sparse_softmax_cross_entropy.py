import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import f1_score
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# x_train = np.array(x_train).reshape(x_train.shape[0],-1)
# x_test = np.array(x_test).reshape(x_test.shape[0],-1)

model = tf.keras.Sequential([
keras.layers.Flatten(),
keras.layers.Dense(units=128,activation=tf.nn.relu),
keras.layers.Dense(10, activation = tf.nn.softmax)
])

# loss='sparse_categorical_crossentropy'
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
hist = model.fit(x_train/255,y_train,epochs=10, verbose=0 )
y_pred1  = model.predict(x_test/255)
y_pred1 = np.argmax(y_pred1, axis=1)
results1 = (np.array([y_pred1 == y_test]).astype(int).reshape(-1,1))
acc1 = (sum(results1)/results1.shape[0])[0]

print("case 1: loss='sparse_categorical_crossentropy' : "+ str(hist.history['acc'][9]))
print("calculated test acc : "+ str(acc1))
print("_________________________________________________")
# loss=tf.losses.sparse_softmax_cross_entropy
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.losses.sparse_softmax_cross_entropy,
              metrics=['accuracy'])
hist = model.fit(x_train/255,y_train,epochs=10,verbose=1)
y_pred2  = model.predict(x_test/255)
y_pred2 = np.argmax(y_pred2, axis=1)
results2 = (np.array([y_pred2 == y_test]).astype(int).reshape(-1,1))
acc2 = (sum(results2)/results2.shape[0])[0]
print("case 2: loss=tf.losses.sparse_softmax_cross_entropy : "+ str(hist.history['acc'][9]))
print("calculated test acc : "+ str(acc2))
print("_________________________________________________")
# loss=tf.losses.softmax_cross_entropy
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.losses.softmax_cross_entropy,
              metrics=['accuracy'])
from keras.utils import to_categorical
y_train_onehot = to_categorical(y_train)
hist = model.fit(x_train/255,y_train_onehot,epochs=10,verbose=0)
y_pred3  = model.predict(x_test/255)
y_pred3 = np.argmax(y_pred3, axis=1)
results3 = (np.array([y_pred3 == y_test]).astype(int).reshape(-1,1))
acc3 = (sum(results3)/results3.shape[0])[0]
print("case 3: loss=tf.losses.softmax_cross_entropy : "+ str(hist.history['acc'][9]))
print("calculated test acc : "+ str(acc3))
