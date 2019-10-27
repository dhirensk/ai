#  Dataset downloaded from http://www.statmt.org/europarl/v7/fr-en.tgz

import tensorflow as tf

from sklearn.model_selection import train_test_split
import os
import io
import numpy as np
import re
import unicodedata

def read_file(filename):
    path = os.getcwd()
    path = os.path.join(path, filename)
    file = io.open(path,encoding='UTF-8')
    lines = file.read()
    file.close()
    return lines

#english_file = read_file("europarl-v7.fr-en.en")
#french_file = read_file("europarl-v7.fr-en.fr")

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(s):
    #s = unicode_to_ascii(s.lower().strip())
    s = s.lower()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # cannot do this for non-english letters, can be done for french, spanish, german
    #s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)

    s = s.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    s = '<start> ' + s + ' <end>'
    return s

# split sentences separated by . or new line
# use below when two different files
def create_dataset(file1, file2, num_samples=None):
    x_lines = read_file(file1)
    y_lines = re.split('\n|\.',read_file(file1))[:num_samples]
    x_lines = re.split('\n|\.',read_file(file2))[:num_samples]
    X = [ preprocess_sentence(s) for s in x_lines]
    Y_true = [preprocess_sentence(s) for s in y_lines]
    return X,Y_true

def create_dataset2(filename, num_samples):
    path = os.getcwd()
    path = os.path.join(path, filename)
    file = io.open(path,encoding='UTF-8')
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_samples]]

    return zip(*word_pairs)
#X_text, Y_text = create_dataset("europarl-v7.fr-en.fr","europarl-v7.fr-en.en", num_samples=64)
X_text, Y_text = create_dataset2("hin.txt", num_samples=None)

print(X_text[50])
print(Y_text[50])
# create a function to tokenize words into index using inbuild tokenizer vocabulory
# important to override filter otherwise it will filter out all punctuation, plus tabs and line breaks, minus the ' character.
def tokenize(input):
   tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
   tokenizer.fit_on_texts(input)
   sequences = tokenizer.texts_to_sequences(input)
  # print(max_len(sequences))
   sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
   return  sequences, tokenizer

def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max( len(t) for t in tensor)

# Tokenize each word into index and return the tokenized list and tokenizer
X , X_tokenizer = tokenize(X_text)
Y, Y_tokenizer = tokenize(Y_text)
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

Tx = max_len(X)  #142
Ty = max_len(Y)  #156

X_tokenizer.word_index['<start>'] #'<start>': 2   # tokenize by frequency
input_vocab_size = len(X_tokenizer.word_index)+1  # add 1 for reserve index 0 which is not included in dictionary
output_vocab_size = len(Y_tokenizer.word_index)+ 1

#Generating tf.data.Dataset

BATCH_SIZE = 64
BUFFER_SIZE = len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024

dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
print(example_X.shape)  #(64, 142)
print(example_Y.shape)  #(64, 156)
#Encoder

class Encoder(tf.keras.Model):
    def __init__(self, input_vocab_size, embeddingdims, rnn_units, batch_size):
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.embedding_dims = embeddingdims
        self.gru_units = rnn_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.embedding_dims )
        self.GRU = tf.keras.layers.GRU(self.gru_units, return_sequences= True, return_state=True)

    def initialize_initial_state(self):
        return tf.zeros((self.batch_size, self.gru_units))
    def call(self, inputs, initial_state):
        X = self.embedding(inputs)  # [m,Tx] --> [m,Tx, embedding_dims]
        activations, cell_state = self.GRU(X, initial_state= initial_state)
        return activations, cell_state

encoder = Encoder(input_vocab_size, embedding_dims, rnn_units,BATCH_SIZE)
encoder_initial_cell_state = encoder.initialize_initial_state()
print(encoder_initial_cell_state.shape)   #(64, 1024)
sample_activations, sample_cell_state = encoder(example_X, encoder_initial_cell_state)
print(sample_activations.shape) # (64, 142, 1024)  batch_size, tx, rnn_units
print(sample_cell_state.shape) #(64, 1024)   batch_size, rnn_units


# 2 layer attention network
# attention weights : concatenator [a, s<t-1>] --> Dense(10) [m,tx,10]--> Dense(1)[m,tx,1] -->tanh --> softmax --> alphas
# context vector : dot (alphas , a, axes =1) --> [m,tx,n_a] --> [m,tx]   or aphas * a --> tf.reduce_sum(axis =1)

#Global Attention Model

#The idea of a global attentional model is to consider all the hidden states of the encoder when deriving the context
# vector ct. In this model type, a variable-length alignment vector a<t> (attention_weights), whose size equals the
# number of timesteps on the source side, is derived by comparing the current target hidden state h<t> with each source hidden state ̄hs:

class Attention(tf.keras.Model):
    def __init__(self, dense_units):
        super().__init__()
        self.dense1_a = tf.keras.layers.Dense(dense_units)
        self.dense1_s_prev = tf.keras.layers.Dense(dense_units)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, a, s_prev):
        #Bahdanau's Additive Style
        s_prev = tf.expand_dims(s_prev, axis=1) # s_prev [ m, n_s] --> [m, 1, n_s] #add dimension for tx for compatibility
        score =  self.dense1_s_prev(s_prev) + self.dense1_a(a) # [m, tx, n_s+n_a] --> [m, tx, dense_units]
        score = tf.nn.tanh(self.dense2(score)) #   [m,tx,dense_units] -> [m, tx, 1]
        attention_weights = tf.nn.softmax(score, axis = 1)  # [m,tx,1] (64,142,1)

        #Luong's Multiplicative Style

        #context vectors

        context_weights = attention_weights * a  # [m,tx,n_a]   (64,142,1024)
        context_vector = tf.reduce_sum( context_weights, axis=1)  #[m,tx]   (64,142)
        return attention_weights, context_vector

attention_layer = Attention(1024)
sample_attenion_weights, sample_ctx = attention_layer(sample_activations,sample_cell_state)
print("Sample Attention Weights [m, Tx, 1] -->{}".format(sample_attenion_weights.shape))  # (64, 142, 1)
print("Sample Context Shape [m, n_a] --->{}".format(sample_ctx.shape))   # (64, 1024)

#Decoder
# embedding layer [m, ty, output_vocab_size] + generated context Vector at time <t>--> GRU

class Decoder(tf.keras.Model):

    def __init__(self, output_vocab_size, embedding_dims,rnn_units, batch_size):
        super().__init__()
        self.embedding_layer = tf.keras.layers.Embedding(output_vocab_size, embedding_dims)
        self.attention_layer = Attention(rnn_units)
        self.BahdanauAttention = tf.contrib.
        self.GRU = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(output_vocab_size)

    def call(self, Y, a, s_prev):
        #unlike encoder which will encode all timesteps, the decoder will work on 1 timestep at a time,
        # because context needs to be generated at each step, so input will of shape m,1
        embeddings = self.embedding_layer(Y)  # [m, 1, output_vocab_size] --> [m, 1, embedding_dims]
        attention_weights, context_vector = attention_layer(a, s_prev)
        # decoder timestep receives activations from previous time step which is used to calculate context
        # merge context and embeddings to pass as input to GRU
        # context_vector.shape [m, rnn_units]
        context_vector_reshape = tf.expand_dims(context_vector, axis=1) # [m,1,rnn_units]
        #concatenate context_vector with embeddings for m,T<x> timestep on last dimension
        embedding_context = tf.concat([embeddings, context_vector_reshape], axis=-1)  # [m,1, embedding_dims+ rnn_units]
        activations, cell_state = self.GRU(embedding_context)   #[m,1,rnn_units], [m,rnn_units]
        activations = tf.reshape(activations,[-1, activations.shape[2]])  # [m, rnn_units] # remove dim for tx
        output = self.dense(activations)
        # Use Argmax at inference time
        return output, cell_state

decoder = Decoder(output_vocab_size, embedding_dims,rnn_units, BATCH_SIZE)
sample_input = tf.random.uniform((BATCH_SIZE, 1),minval=0, maxval=output_vocab_size, dtype=tf.int32)  # [64,1]  [between vocab 0..4951]
sample_output,_ = decoder(sample_input, sample_activations, sample_cell_state)
print("Sample Output Shape [m, output_vocab_size] -->{}".format(sample_output.shape))

# Optimizer and Loss Function

optimizer = tf.keras.optimizers.Adam()

def loss_function(y_pred, y):
    # we are running decoder for each time step, Expected shape of y and y_pred
    #shape of y [batch_size, ty] --> [64]
    #shape of y_pred [batch_size, output_vocab_size] --> [64,4951]
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    #skip loss calculation for padding i.e. y = 0 index is reserved for padding
    # y is a tensor of batch_size,1 . Create a mask when y=0
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss


#training using teacher forcing
#Creates a callable TensorFlow graph from a Python function.
#function constructs a callable that executes a TensorFlow graph (tf.Graph) created by tracing the TensorFlow operations in func.
#This allows the TensorFlow runtime to apply optimizations and exploit parallelism in the computation defined by func.

def train_step(input_batch, output_batch,encoder_initial_cell_state):
    with tf.GradientTape() as tape:

        #initialize loss = 0
        loss = 0
        # we can do initialization in outer block
        #encoder_initial_cell_state = encoder.initialize_initial_state()
        a, c_tx = encoder(input_batch, encoder_initial_cell_state)
        #pass encoder memory_state as input to decoder
        s_prev = c_tx
        # decoder input starts with index of <start>
        # decoder Y_true starts with output_batch[:,t]

        #take first timestap which starts with <start> and pass it decoder. We can either construct it or take from output_batch
        #decoder_input = tf.expand_dims([Y_tokenizer.word_index['<start>']]* BATCH_SIZE,1)  #64,1
        decoder_input = tf.expand_dims(output_batch[:, 0],1)
        # pass the activations to decoder network
        for t in range(1,Ty):

            #new cellstates are passed as s_prev to decoder for next time step
            prediction, s_prev = decoder(decoder_input, a, s_prev)

            #Calculate loss
            loss += loss_function(prediction, output_batch[:,t])
            #teacher forcing lets you input actual input at t and expects to predict output t+1
            decoder_input = tf.expand_dims(output_batch[:, t],1)

        batch_loss = loss/Ty
        #Returns the list of all layer variables / weights.
        variables = encoder.variables + decoder.variables
        # differentiate loss wrt variables
        gradients = tape.gradient(loss, variables)

        #grads_and_vars – List of(gradient, variable) pairs.
        grads_and_vars = zip(gradients,variables)
        optimizer.apply_gradients(grads_and_vars)
        return batch_loss

# training
#get existing checkpoint objects
#Object based Checkpointing
checkpointdir = os.path.join(os.getcwd(),"nmt_logs")
chkpoint_prefix = os.path.join(checkpointdir, "chkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)

checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoder = encoder, decoder = decoder)
try:
    checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
except:
    print("No checkpoint found at {}".format(checkpointdir))


# in tf 1.4 the enumerate(dataset) goes into infinite loop.
epochs = 2
for i in range(1, epochs+1):

    encoder_initial_cell_state = encoder.initialize_initial_state()
    total_loss = 0.0


    for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
        total_loss += batch_loss
        if (batch+1)%20 == 0:
            print("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), i, batch+1))
            checkpoint.save(file_prefix = chkpoint_prefix)
    #for (batch, (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
    #    batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
    #    total_loss += batch_loss
    #    print("total loss: {} epoch {} ".format(total_loss, i))

#Inference
#Create input sequence to pass to encoder.
# The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
#Stop predicting when the model predicts the end token.
#And store the attention weights for every time step.

input_raw = "Welcome Customers.\nYour car is ready."
# def inference(input_raw):
input_lines = input_raw.split("\n")
# We have a transcript file containing English-Hindi pairs
# Preprocess X
input_lines = [preprocess_sentence(line) for line in input_lines]
input_sequences = [[X_tokenizer.word_index[w] for w in line.split(' ')] for line in input_lines]
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=Tx, padding='post')
print(input_sequences.shape)
output_sequences = []

# iterate for each row of input
for i in range(len(input_sequences)):
    output_line = []
    inp = tf.convert_to_tensor(tf.expand_dims(input_sequences[i], axis=0))
    print(inp.shape)  # [1, tx]
    encoder_initial_cell_state = tf.zeros((1, rnn_units))  # batch size is 1 line of input
    print(encoder_initial_cell_state.shape)
    a, c_tx = encoder(inp, encoder_initial_cell_state)
    print("a :", a.shape)
    print("c_tx : ", c_tx.shape)
    decoder_input = tf.expand_dims([Y_tokenizer.word_index['<start>']], axis=0)  # [batch_size=1, ty=1]
    print(decoder_input.shape)

    # propagte through timesteps in decoder
    s_prev = c_tx
    for j in range(Ty):
        prediction, cell_state = decoder(decoder_input, a, s_prev)
        # print(prediction.shape)  [1, 3025]

        assert prediction.shape == (1, output_vocab_size)
        #  (1, 19803)--> prediction[0] --> (19803,)
        prediction_index = tf.argmax(prediction[0]).numpy()
        #pass predicted output as decoder input for next time step
        decoder_input = tf.expand_dims([prediction_index], axis=0)  # [1,1]
        prediction_word = Y_tokenizer.index_word[prediction_index]
        # print(prediction[0].numpy())
        output_line.append(prediction_word)
        if prediction_word == '<end>':
            break

    output_sequences.append(output_line)

print(input_raw,"\n")
for i in range(len(output_sequences)):
    output =  ' '.join([w for w in  output_sequences[i][:-1]])
    print(output)

#Welcome Customers.
#Your car is ready.

#आपका स्वागत है।
#गाड़ी तैयार है।