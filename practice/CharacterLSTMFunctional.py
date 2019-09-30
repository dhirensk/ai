#[1]  Import Libararies

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import re
import os
from tensorflow.python.keras.utils import to_categorical

#[2]  Import dataset
url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
file = tf.keras.utils.get_file('shakespeare.txt', url)
data = open(file, 'r').read()
chars = np.array(sorted(set(data)))

#[3]  Build Vocabulary
chr_to_idx = { chr:idx for idx, chr in enumerate(chars)}
# idx_to_chr = dict([(idx, chr) for (chr,idx) in chr_to_idx.items()])
idx_to_chr = { idx:chr for idx,chr in enumerate(chars)}

#character to integer mapping using vocabulary
chr_to_int = np.array([chr_to_idx[c] for c in data])


#[4] Initialization
Tx = 100   #number of sequences in each example
n_a = 1024   #number of hidden layers in RNN
n_values = len(chr_to_idx)  #65 Number of unique characters
n_chars = len(data)    # 1115394 Number of characters in dataset
n_m = n_chars//Tx    # 11153 number of examples per epoch, we will discard the remainder


#[5] Prepare Dataset Batches


char_dataset = tf.data.Dataset.from_tensor_slices(chr_to_int)
sequences = char_dataset.batch(Tx + 1, drop_remainder=True)

# only with eager execution
#for item in sequences.take(5):
#    print(repr( ''.join(chars[item.numpy()])))

#'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '
#'are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you k'
#"now Caius Marcius is chief enemy to the people.\n\nAll:\nWe know't, we know't.\n\nFirst Citizen:\nLet us ki"
#"ll him, and we'll have corn at our own price.\nIs't a verdict?\n\nAll:\nNo more talking on't; let it be d"
#'one: away, away!\n\nSecond Citizen:\nOne word, good citizens.\n\nFirst Citizen:\nWe are accounted poor citi'

#Create mapfunction to generate X and Y by taking batch and shifting it by 1 character
#      "Hello World " 12
#X --> "Hello World"  11  input[:-1] take out last character
#Y --> "ello World "  11  input[1:]  take out first character

def generateXY(seq):
    X = seq[:-1]
    Y = seq[1:]
    return X, Y

dataset = sequences.map(generateXY)

#for input_x, target_y in dataset.take(1):
#    print( repr(''.join(chars[input_x])))
#    print(repr(''.join(chars[target_y])))

#'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou'
#'irst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '

#We used tf.data to split the text into manageable sequences. But before feeding this data into the model,
# we need to shuffle the data and pack it into batches.
BATCH_SIZE = 64
steps_per_epoch = n_m // BATCH_SIZE
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 10000
currdir = os.getcwd()
logsdir = os.path.join(currdir,"logs")
if not os.path.exists(logsdir):
    os.makedirs(logsdir)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True).repeat()
print("Dataset Shape \n", dataset.output_shapes)
#(TensorShape([Dimension(64), Dimension(100)]), TensorShape([Dimension(64), Dimension(100)]))


#[6]   Create Model Class
vocab_size = n_values
embedding_dims = 256


#input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
#output_dim: int >= 0. Dimension of the dense embedding.
#Input shape :2D tensor with shape: (batch_size, sequence_length).
#Output shape: 3D tensor with shape: (batch_size, sequence_length, output_dim).

#If a RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors:
#- If using a Sequential model, specify the batch size by passing a `batch_input_shape` argument to your first layer.
#- If using the functional API, specify the batch size by passing a `batch_shape` argument to your Input layer.
class SequenceModel():
    def __init__(self, mode, logsdir, vocab_size, embedding_dims, batch_size, n_a):
        self.epoch = 0
        self.mode = mode
        self.logsdir = logsdir
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.n_a = n_a
        self.embedding_dims = embedding_dims
        self.checkpointpath = self.setchkpointdir()
        self.sequencemodel = self.build(self.vocab_size, self.embedding_dims,self.batch_size, self.n_a)

    def build(self, vocab_size,embedding_dims,  batch_size, n_a):
        # Embedding object is not subscriptable in functional model if its the first layer
        #add an input layer
        # Dataset.take(1).shape (TensorShape([Dimension(64), Dimension(100)]), TensorShape([Dimension(64), Dimension(100)]))
      #  input = tf.cast(tf.keras.Input(shape=[Tx,]), tf.float32) not permitted
        input = tf.keras.Input(shape=(Tx,), dtype= tf.float32)
        embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dims, batch_input_shape=[batch_size, None],
                                  trainable=True, name='embed')(input)
        lstm_cell = tf.keras.layers.LSTM(n_a, return_sequences= True, stateful=False, name='lstm')(embedding)
        output = tf.keras.layers.Dense(vocab_size, name='output')(lstm_cell)
        return tf.keras.Model(inputs = input, outputs = output)


    def compile(self, optimizer, loss_function):
        self.sequencemodel.compile(optimizer=optimizer, loss=loss_function)

    def load_weights(self, weights=None):

        if weights=="last":
            checkpoints = [f for f in os.listdir(self.logsdir) if os.path.isfile(os.path.join(self.logsdir, f))]
            print(checkpoints)
            if len(checkpoints)> 0:
                checkpoints = filter(lambda f: f.startswith("LSTM_character_model_"), checkpoints)
                #filter operation returns an iterator, length of an interator cannot be determined using len

                if checkpoints:
                    checkpoint = sorted(checkpoints)[-1]  #get last checkpoint
                    print(checkpoint)
                    #regex = r".*[/\\][\w]LSTM_character_model_+(\d{3})\.h5"
                    regex = "LSTM_character_model_+(\d{3})\.h5"
                    m = re.match(regex, checkpoint)
                    if m:
                        # Epoch number in file is 1-based, and in Keras code it's 0-based.
                        # So, adjust for that then increment by one to start from the next epoch
                        print("Last Checkpoint file: ",m.string)
                        self.epoch = int(m.group(1)) - 1 + 1
                        #group 0 returns entire match, group 1 returns value in first group()
                        self.model_path = os.path.join(logsdir,checkpoint)
                        self.sequencemodel.load_weights(self.model_path)
                        print("starting from epoch: ", self.epoch+1)
                    else:
                        self.epoch = 0

    def train(self, epochs, weights=None):

        self.load_weights(weights)  #load last checkpoint weight if weights = last
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpointpath, save_weights_only=True)
        self.sequencemodel.fit(dataset.make_one_shot_iterator(), epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[model_checkpoint],
                  initial_epoch=self.epoch)
        #set the current epoch to max after training ends
        self.epoch = max(self.epoch, epochs)

    def setchkpointdir(self):
            # windows C:\\path\\to\\logsdir\\LSTM_character_model_001.h5
            # unix /path/to/logsdir//LSTM_character_model_001.h5
        return  os.path.join(logsdir, "LSTM_character_model_{epoch:03d}.h5")


    def prediction(self, input):
        output_prediction = self.sequencemodel(input)
        #print(output_prediction.shape)
        #(64, 100, 65)  # (batch_size, Tx, vocab_size)

        #input_x, target_y  =  dataset.take(1)

        #ValueError: not enough values to unpack (expected 2, got 1)


        sample_example = output_prediction[0] #first element 1,100,65
        #print(sample_example.shape)   # 100,65

        #logits – 2-D Tensor with shape `[batch_size, num_classes]`. Each slice `[i, :]` represents the unnormalized log-probabilities for all classes.
        #num_samples – 0-D. Number of independent samples to draw for each row slice.
        sample_indices = tf.random.categorical(sample_example, num_samples=1)  # shape 100,1
        sample_indices = tf.squeeze(sample_indices, axis=-1).numpy()  #shape 100,

        sample_chars = repr(''.join(chars[sample_indices]))
        return sample_chars

# train/inference
model = SequenceModel("train",logsdir,vocab_size,embedding_dims, BATCH_SIZE, n_a)

#[10] Prediction before training model
print(dataset.take(1).output_shapes)
# (TensorShape([Dimension(64), Dimension(100),Dimension(65)]), TensorShape([Dimension(64), Dimension(100),Dimension(65)]))
for input_x, target_y in dataset.take(1):

    print(input_x.shape)
    print(target_y.shape)
    # (64, 100)
    # (64, 100)
    print( input_x[0])
    # Print first record (batchsize = 1)

    input_chars = repr(''.join(chars[input_x[0]]))  # 100,
    print("Checking sequence generation before training")
    print("Input Sample \n\n",input_chars)
    print("\n")
    print("Output Sequence")

    output_sample = model.prediction(input_x)
    print(output_sample)

# Because our model returns logits, we need to set the from_logits flag.(We dont have any activations defined in dense.)
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer,loss)
model.train(6,weights=None)  #epochs, weights = "last"/None


#Already trained model
model.load_weights(weights="last")

#[11] Prediction after training
#prediction("After Training")
print("Checking sequence generation after training")
for input_x, target_y in dataset.take(1):

    print(input_x.shape)
    print(target_y.shape)
    # (64, 100)
    # (64, 100)
    # Print first record (batchsize = 1)
    input_chars = repr(''.join(chars[input_x[0]]))  # 100,
    print("Input Sample \n\n",input_chars)
    print("\n")
    print("Output Sequence")

    output_sample = model.prediction(input_x)
    print(output_sample)

#[11] Sampling from some starting sequence

# Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.
# To run the model with a different batch_size, we need to rebuild the model and restore the weights from the checkpoint.
#BATCH_SIZE = 1
model_sampling = SequenceModel("train",logsdir,vocab_size, embedding_dims, 1,n_a)
model_sampling.load_weights("last")
model_sampling.sequencemodel.build(tf.TensorShape([1,None]))

#Builds the model based on input shapes received.
#This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.
#This method only exists for users who want to call model.build() in a standalone way (as a substitute for calling the model
# on real data to build it). It will never be called by the framework (and thus it will never throw unexpected errors
# in an unrelated workflow).
model_sampling.sequencemodel.summary()

def sample_sequence(start_string, sample_length,model):
    # convert text to int on sample
    start_string_int = [chr_to_idx[chr] for chr in start_string]
    start_string_int =  tf.expand_dims(start_string_int, axis=0)
    print("Shape of start_string : ", start_string_int, start_string_int.shape)
    sampled_sequence = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    #batch_size =1 since we are passing single start_string
    # reset states
    model.sequencemodel.reset_states()
    for i in range(sample_length):
        predictions = model.sequencemodel([start_string_int])  #(1,5,65) batch=1, tx=5, vocab_size=65
        #print(predictions.shape)
        predictions = tf.squeeze(predictions, axis=0) # remove batch=1 dimension  (5,65)
        predictions = predictions/temperature
        last_predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()  #(5,1) --> take last element of 5 ids
        #print(last_predicted_id.shape)  #shape is ()
        start_string_int = tf.expand_dims([last_predicted_id], axis=0)    #(1,) -->  (1,1)
        sampled_sequence.append(chars[start_string_int])
    return start_string+''.join(sampled_sequence)

sampled_sequence = sample_sequence(u"hello",100, model_sampling)
print(sampled_sequence)


