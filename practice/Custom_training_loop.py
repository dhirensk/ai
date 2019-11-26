import tensorflow as tf
import numpy as np
#tf.enable_eager_execution()

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#Add channels for GrayScale
X_train = np.expand_dims(X_train, axis = -1).astype(np.float32)
X_test = np.expand_dims(X_test, axis = -1).astype(np.float32)
X_train /= 255
X_test /= 255
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

def loss_function(y_true,y_pred):
    # applied softmax in output dense, so output is probability densities instead of logits
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss = ce(y_true=y_true, y_pred=y_pred)
    loss = tf.reduce_mean(loss)
    return loss

total_samples = X_train.shape[0]
BUFFER_SIZE = total_samples
BATCH_SIZE = 64
steps_per_epoch = total_samples // BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        variables = model.variables
        # y should be a 1 column vector [ batch_size]
        # x should be  [ batch_size, classes]
        prediction = model(x)
        batch_loss = loss_function(y, prediction)
        gradients = tape.gradient(batch_loss, variables )
        grads_and_vars = zip(gradients, variables)
        optimizer.apply_gradients(grads_and_vars)
        return batch_loss

# train on epochs

epochs = 10
for i in range(1, epochs+1):
    total_loss  = 0
    for batch, (x_in, y_in) in enumerate( dataset.take(steps_per_epoch)):
       batch_loss = train_on_batch(x_in,y_in)
       total_loss+= batch_loss
       if batch % 30 ==0:
           print( "batch : {} batch_loss {}".format(batch, batch_loss.numpy()))
    print(" epoch : {} total loss :{}".format(i, total_loss.numpy()))


# prediction
from sklearn.metrics import confusion_matrix
prediction = tf.argmax(model(X_test), axis = 1)
cm = confusion_matrix(Y_test, prediction)
print(cm)

# [[ 974    0    1    1    0    1    2    0    1    0]
#  [   0 1128    0    3    2    0    1    0    1    0]
#  [   1    5 1015    1    1    0    0    8    1    0]
#  [   0    0    2 1004    0    3    0    0    1    0]
#  [   0    0    0    0  977    0    2    0    0    3]
#  [   1    0    1    7    0  881    1    1    0    0]
#  [   3    2    0    1    1    6  943    0    2    0]
#  [   0    4    5    1    0    0    0 1017    1    0]
#  [   4    0    2    0    2    2    2    3  957    2]
#  [   0    3    2    4   12    9    0    6    5  968]]
