import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
global_init = tf.global_variables_initializer()

logdir = os.path.join(os.getcwd() + "tensorboard")

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=True, )
mnist_dset = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist_dset.load_data()
x_train = x_train[0:10000]
y_train = y_train[0:10000]
K = tf.keras.backend
if K.image_data_format() == 'channels_first':
    K.set_image_data_format('channels_last')
assert K.image_data_format() == 'channels_last'
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = np.expand_dims(x_train,axis=-1)
x_test = np.expand_dims(x_test,axis=-1)

assert x_train[0].shape == (28,28,1)
assert x_test[0].shape == (28,28,1)


y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10)
assert y_train[0].shape == (10,)
assert y_test[0].shape == (10,)

class Conv2dActivation(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        model = self.model



class MnistModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu', name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', name = 'conv2')
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128,activation='relu')
        self.dense2 = tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax)

    def call(self, inputs):
        X = self.conv1(inputs)
        X = self.conv2(X)
        X = self.maxpool1(X)
        X = self.dropout1(X)
        X = self.flatten1(X)
        X = self.dense1(X)
        X = self.dense2(X)
        return X

mnist_model = MnistModel()

mnist_model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
mnist_model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=1,batch_size=128 , callbacks=[Conv2dActivation()])

class Conv1_Activations(MnistModel):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        X = self.conv1(inputs)
        return X


conv1_model = Conv1_Activations()
sample_input = np.expand_dims(x_train[0],axis=0)
assert sample_input.shape == (1,28,28,1)
conv1_activations = conv1_model.predict(sample_input)
columns = 4
nbf= 32
#conv1 has 32 filters. Display in grid 8X4
for i in range(32):
    plt.subplot(nbf/columns, columns,i+1) #index from 1 in subplot
    plt.imshow(conv1_activations[0,:,:,i])


