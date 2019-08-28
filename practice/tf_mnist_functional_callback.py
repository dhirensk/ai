import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

logdir = os.getcwd() + "\\tensorboard"

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


sample_input = np.expand_dims(x_train[0],axis=0)
assert sample_input.shape == (1,28,28,1)

conv2_activations = np.empty(shape=(1,24,24,0), dtype=np.float32)  # use first dim for epochs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_images=True)
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        input_layer = self.model.input
        output_layer = self.model.get_layer(name="conv2")
        conv2_model = tf.keras.Model(inputs=input_layer, outputs=output_layer.output)
        activations = conv2_model.predict(sample_input)
        print(activations.shape)
        global conv2_activations
        if conv2_activations.shape[3] == 0:
            conv2_activations = activations
        else:
            conv2_activations = np.append(conv2_activations, activations, axis= 0)


input = tf.keras.Input(shape=(28,28,1))
def build_conv_layers(input):

    X = tf.keras.layers.Conv2D(32,(3,3),activation='relu', name="conv1")(input)
    X = tf.keras.layers.Conv2D(64,(3,3), activation='relu', name ="conv2")(X)
    X = tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = tf.keras.layers.Dropout(0.2)(X)

    return X

def build_out_layer(conv2d):
    output = tf.keras.layers.Flatten()(conv2d)
    output = tf.keras.layers.Dense(128,activation='relu')(output)
    output = tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax)(output)
    return output

conv = build_conv_layers(input)
output = build_out_layer(conv)
mnist_model = tf.keras.Model(inputs = input, outputs = output)

mnist_model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
mnist_model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=10,batch_size=128 , callbacks=[MyCustomCallback()])

columns = 8
nbf = 64
# conv1 has 32 filters. Display in grid 8X4
epochs = 10
for epoch in range(1,epochs+1,1):
    if (epoch) %5 ==0:
        fig = plt.figure()
        fig.suptitle("Conv2D Activations for epoch : "+ str(epoch))
        for i in range(64):
            ax1 = plt.subplot(nbf / columns, columns, i + 1)  # index from 1 in subplot
            plt.subplots_adjust(hspace=0.5)
            ax1.set_title("Kernel : " +str(i+1), fontsize=6)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            plt.imshow(conv2_activations[epoch-1, :, :, i], cmap='gray')