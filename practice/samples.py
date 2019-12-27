import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

assert isinstance(train_dataset, tf.data.Dataset)

train_dataset.element_spec
#tfds.core.SplitInfo num_examples
train_size = info.splits['train'].num_examples
test_size = info.splits['test'].num_examples
batch_size = 32
epochs = 10
steps_per_epoch = train_size//batch_size
validation_steps = test_size//batch_size

train_dataset = train_dataset.map( lambda x,y: (tf.cast(x,tf.float32)/255.0,y ))
test_dataset = test_dataset.map( lambda x,y: (tf.cast(x,tf.float32)/255.0,y ))

train_dataset = train_dataset.shuffle(buffer_size= 1024).batch(batch_size=32, drop_remainder=True)
test_dataset = test_dataset.shuffle(buffer_size= 1024).batch(batch_size=32, drop_remainder=True)
for img, label in train_dataset.take(1):
    print(img[0].shape)
    print(label[0])

class MnistClassifier(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.flatten_ = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units = 32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units = 32, activation= 'relu')
        self.output_ = tf.keras.layers.Dense(units = self.num_classes, activation='softmax')

    def call(self, inputs, training=True, mask=None):
        x = self.flatten_(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_(x)
        return x

model = MnistClassifier(num_classes=10)
optimizer = tf.keras.optimizers.Adam()

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss_function(true_label, logits):
    #print(true_label.shape)
    #y_true = tf.keras.utils.to_categorical(true_label,num_classes=10)
    #cc = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = loss_obj(y_true=true_label,y_pred=logits)
    loss = tf.reduce_mean(loss)
    return loss


ut_logits = tf.constant([0.,0.,0.90,0.07,0.,0.01,0.,-0.03,0.,0.])
ut_true = tf.constant([2])
print(loss_function( ut_true,ut_logits))


for _,(img, label) in enumerate(train_dataset.take(1)):
   # print(img.shape)
    #print(label.shape)
    pred = model(img)
    #print(pred)
    print(loss_function(label,pred ))


@tf.function
def one_training_step(input_batch, true_y):
    loss = 0.0
    with tf.GradientTape() as tape:
        pred_y = model(input_batch)
        loss += loss_function(true_y, pred_y)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

## training

for i in range(1,epochs+1):
    total_loss = 0.0
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for step, (input_batch, true_y) in enumerate(train_dataset.take(steps_per_epoch)):
        #print(input_batch.dtype)
        #print(true_y.dtype)
        batch_loss = one_training_step(input_batch, true_y)
        total_loss+= batch_loss
        epoch_loss_avg(batch_loss)  # Add current batch loss
        # Compare predicted label to actual label
        epoch_accuracy(true_y, model(input_batch))

        if step % 50 == 0:
            continue
            print("Batch loss at epoch {} : step {} is {}  Accuracy {}".format(i,step, epoch_loss_avg.result(), epoch_accuracy.result()))
    print("Total loss at end of epoch {} is {}".format(i, total_loss/steps_per_epoch))


