

import tensorflow as tf
import tensorflow_addons as tfa
print(tf.__version__)
from sklearn.model_selection import train_test_split
import os
import io
import numpy as np
import re
import unicodedata
import urllib3
import shutil
import zipfile
import itertools
from google.colab import drive

"""### Download File"""

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)
print(zipfilename)
with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

"""### Preprocess File"""

def read_file(filename):
    path = os.getcwd()
    path = os.path.join(path, filename)
    file = io.open(path,encoding='UTF-8')
    lines = file.read()
    file.close()
    return lines

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# For non-english characterset translations such as Hindi, Russian, etc. we keep unicode. 
def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
    s = s.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)

    s = s.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    s = '<start> ' + s + ' <end>'
    return s

"""### Create Dataset"""

def create_dataset(filename, num_samples):
    path = os.getcwd()
    path = os.path.join(path, filename)
    file = io.open(path,encoding='UTF-8')
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_samples]]

    return zip(*word_pairs)

X_text,Y_text,_  = create_dataset("fra.txt", num_samples=5000)

print(X_text[4000:4001])
print(Y_text[4000:4001])

#total samples
print("Total Samples : ", len(X_text))

# create a function to tokenize words into index using inbuild tokenizer vocabulory
# important to override filter otherwise it will filter out all punctuation,
# plus tabs and line breaks, minus the ' character.
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

Tx = max_len(X)
Ty = max_len(Y)

print("Max length English sentence denoted as Tx : ", Tx)
print("Max length French sentence denoted as Ty: ", Ty)

X_tokenizer.word_index['<start>'] #'<start>': 2   # tokenize by frequency
input_vocab_size = len(X_tokenizer.word_index)+1  # add 1 for 0 sequence character
output_vocab_size = len(Y_tokenizer.word_index)+ 1
print("input_vocab_size : ", input_vocab_size)
print("output_vocab_size : " ,output_vocab_size)

"""### Model Parameters"""

BATCH_SIZE = 64
BUFFER_SIZE = len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32   #used to initialize DecoderCell Zero state

dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
print(example_X.shape) 
print(example_Y.shape)

dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
print(example_X.shape) 
print(example_Y.shape)

"""### Creating Encoder-Decoder Model based on tfa.seq2seq module

### Define Model

The encoder network consists of an encoder embedding layer and a LSTM layer.

The decoder network encompasses both decoder and attention mechanism.

The example uses LuongAttention.
"""

#ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     return_state=True )
    
#DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units,None,BATCH_SIZE*[Tx])
        self.rnn_cell =  self.build_rnn_cell(BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, 
                                          memory_sequence_length=memory_sequence_length)
        #return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size ):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=dense_units)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state



encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size,embedding_dims, rnn_units)

"""### Optimizer and Custom Loss Function"""

optimizer = tf.keras.optimizers.Adam()

"""Here, mask is a zero-one matrix of the same size as decoder_outputs. It masks padding positions outside of the target sequence lengths with values 0."""

def loss_function(y_pred, y):
   
    #shape of y [batch_size, ty]
    #shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    #skip loss calculation for padding sequences i.e. y = 0 
    #[ <start>,How, are, you, today, 0, 0, 0, 0 ....<end>]
    #[ 1, 234, 3, 423, 3344, 0, 0 ,0 ,0, 2 ]
    # y is a tensor of [batch_size,Ty] . Create a mask when [y=0]
    # mask the loss when padding sequence appears in the output sequence
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss

"""To begin with, attention mechanism is initialized without memory."""

decoderNetwork.attention_mechanism.memory_initialized

"""### One step of training on a batch using Teacher Forcing technique"""

def train_step(input_batch, output_batch,encoder_initial_cell_state):
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                        initial_state =encoder_initial_cell_state)

        #[last step activations,last memory_state] of encoder passed as input to decoder Network
        s_prev = [a_tx, c_tx]

        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:,:-1] # ignore <end>
        #compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:,1:] #ignore <start>


        # Decoder Embeddings
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,
                                                                           encoder_state=s_prev,
                                                                           Dtype=tf.float32)
        
        #BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                               sequence_length=BATCH_SIZE*[Ty-1])

        logits = outputs.rnn_output
        #Calculate loss

        loss = loss_function(logits, decoder_output)

    #Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    #grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss

"""### Training
get existing checkpoint objects

Object based Checkpointing
"""

# mount gdrive containing trained checkpoint objects
drive.mount('/content/drive', force_remount=True )

"""We load from previously saved checkpoints from Google Drive if already trained."""

checkpointdir = os.path.join('/content/drive/My Drive/DL',"nmt_tfa_logs_eng_to_fra_withAttention")
chkpoint_prefix = os.path.join(checkpointdir, "chkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)

checkpoint = tf.train.Checkpoint(optimizer = optimizer, encoderNetwork = encoderNetwork, 
                                 decoderNetwork = decoderNetwork)

try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
except:
    print("No checkpoint found at {}".format(checkpointdir))

#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
        return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]

epochs = 1
for i in range(1, epochs+1):

    encoder_initial_cell_state = initialize_initial_state()
    total_loss = 0.0


    for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
        total_loss += batch_loss
        if (batch+1)%20 == 0:
            print("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), i, batch+1))
            checkpoint.save(file_prefix = chkpoint_prefix)

"""### Inference
Create input sequence to pass to encoder.

The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.

Stop predicting when the model predicts the end token.

And store the attention weights for every time step.
"""

#if trained in same session else use checkpoint variable
#decoder_embedding_matrix = tf.train.load_variable(checkpointdir, 'decoderNetwork/decoder_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE')
decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
print(decoderNetwork.decoder_embedding.variables[0].shape)

"""if restoring from checkpoint, lets print all variables related to decoder_embeddings and then select and load the right variable containing decoder embeddings"""

[print(var) for var in tf.train.list_variables(
    checkpointdir) if re.match(r'.*decoder_embedding.*',var[0])]

decoder_embedding_matrix = tf.train.load_variable(
    checkpointdir, 'decoderNetwork/decoder_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE')
print(decoder_embedding_matrix.shape)

#use with scope /cpu:0 for inferencing
#restore from latest checkpoint for inferencing
input_raw="Hi  \nHow are you today"
#input_raw="Wow!"  #checking translation on training set record
#def inference(input_raw):
input_lines = input_raw.split("\n")
# We have a transcript file containing English-Hindi pairs
# Preprocess X
input_lines = [preprocess_sentence(line) for line in input_lines]
input_sequences = [[X_tokenizer.word_index[w] for w in line.split(' ')] for line in input_lines]
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                maxlen=Tx, padding='post')
inp = tf.convert_to_tensor(input_sequences)
#print(inp.shape)
inference_batch_size = input_sequences.shape[0]
encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                              tf.zeros((inference_batch_size, rnn_units))]
encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                initial_state =encoder_initial_cell_state)
#pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
s_prev = [a_tx, c_tx]
#output_sequences = []
print('a_tx :',a_tx.shape)
print('c_tx :', c_tx.shape)
print("s_prev = [a_tx, c_tx] :",np.array(s_prev).shape)


start_tokens = tf.fill([inference_batch_size],Y_tokenizer.word_index['<start>'])
#print(start_tokens)
end_token = Y_tokenizer.word_index['<end>']

greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
#finished,start_inputs = greedy_sampler.initialize(decoder_embedding_matrix,start_tokens,end_token)
#print(finished.shape, start_inputs.shape)

decoder_input = tf.expand_dims([Y_tokenizer.word_index['<start>']]* inference_batch_size,1)
decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

decoder_instance = tfa.seq2seq.BasicDecoder(cell = decoderNetwork.rnn_cell, sampler = greedy_sampler,
                                            output_layer=decoderNetwork.dense_layer)
decoderNetwork.attention_mechanism.setup_memory(a)
decoder_initial_state = decoderNetwork.build_decoder_initial_state(inference_batch_size,
                                                                   encoder_state=s_prev,
                                                                   Dtype=tf.float32)
print("\nCompared to simple encoder-decoder without attention, the decoder_initial_state \
 is an AttentionWrapperState object containing s_prev tensors and context and alignment vector \n ")
print("decoder initial state shape :",np.array(decoder_initial_state).shape)
print("decoder_initial_state tensor \n", decoder_initial_state)

# Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
# One heuristic is to decode up to two times the source sentence lengths.
maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)

#initialize inference decoder

(first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                             start_tokens = start_tokens,
                             end_token=end_token,
                             initial_state = decoder_initial_state)
#print( first_finished.shape)
print("\nfirst_inputs returns the same decoder_input i.e. embedding of  <start> :",first_inputs.shape)
print("start_index_emb_avg ", tf.reduce_sum(tf.reduce_mean(first_inputs, axis=0))) # mean along the batch

inputs = first_inputs
state = first_state  
predictions = np.empty((inference_batch_size,0), dtype = np.int32)                                                                             
for j in range(maximum_iterations):
    outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
    inputs = next_inputs
    state = next_state
    outputs = np.expand_dims(outputs.sample_id,axis = -1)
    predictions = np.append(predictions, outputs, axis = -1)

"""Discard translations on encountering first sequence \<end\>"""

print("English Sentence:")
print(input_raw)
print("\nFrench Translation:")
for i in range(len(predictions)):
    line = predictions[i,:]
    seq = list(itertools.takewhile( lambda index: index !=2, line))
    print(" ".join( [Y_tokenizer.index_word[w] for w in seq]))

"""### Inference using Beam Search with beam_width = 3"""

beam_width = 3
#use with scope /cpu:0 for inferencing
#restore from latest checkpoint for inferencing
input_raw="Hi  \nHow are you today"
#input_raw="Wow!"  #checking translation on training set record
#def inference(input_raw):
input_lines = input_raw.split("\n")
# We have a transcript file containing English-Hindi pairs
# Preprocess X
input_lines = [preprocess_sentence(line) for line in input_lines]
input_sequences = [[X_tokenizer.word_index[w] for w in line.split(' ')] for line in input_lines]
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                maxlen=Tx, padding='post')
inp = tf.convert_to_tensor(input_sequences)
#print(inp.shape)
inference_batch_size = input_sequences.shape[0]
encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                              tf.zeros((inference_batch_size, rnn_units))]
encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                initial_state =encoder_initial_cell_state)
#pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
s_prev = [a_tx, c_tx]
#output_sequences = []



start_tokens = tf.fill([inference_batch_size],Y_tokenizer.word_index['<start>'])
#print(start_tokens)
end_token = Y_tokenizer.word_index['<end>']



decoder_input = tf.expand_dims([Y_tokenizer.word_index['<start>']]* inference_batch_size,1)
decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)


#From official documentation
#NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:

#The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
#The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
#The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.
encoder_memory = tfa.seq2seq.tile_batch(a, beam_width)
decoderNetwork.attention_mechanism.setup_memory(encoder_memory)
print("beam_with * [batch_size, Tx, rnn_units] :  3 * [2, Tx, rnn_units]] :", encoder_memory.shape)
#set decoder_inital_state which is an AttentionWrapperState considering beam_width
decoder_initial_state = decoderNetwork.rnn_cell.get_initial_state(batch_size = inference_batch_size* beam_width,dtype = Dtype)
encoder_state = tfa.seq2seq.tile_batch(s_prev, multiplier=beam_width)
decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 

decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoderNetwork.rnn_cell,beam_width=beam_width,
                                                 output_layer=decoderNetwork.dense_layer)


# Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
# One heuristic is to decode up to two times the source sentence lengths.
maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)

#initialize inference decoder

(first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                             start_tokens = start_tokens,
                             end_token=end_token,
                             initial_state = decoder_initial_state)
#print( first_finished.shape)
print("\nfirst_inputs returns the same decoder_input i.e. embedding of  <start> :",first_inputs.shape)

inputs = first_inputs
state = first_state  
predictions = np.empty((inference_batch_size, beam_width,0), dtype = np.int32)
beam_scores =  np.empty((inference_batch_size, beam_width,0), dtype = np.float32)                                                                            
for j in range(maximum_iterations):
    beam_search_outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
    inputs = next_inputs
    state = next_state
    outputs = np.expand_dims(beam_search_outputs.predicted_ids,axis = -1)
    scores = np.expand_dims(beam_search_outputs.scores,axis = -1)
    predictions = np.append(predictions, outputs, axis = -1)
    beam_scores = np.append(beam_scores, scores, axis = -1)
print(predictions.shape) 
print(beam_scores.shape)

print("-----------------")
print("English Sentence:")
print(input_raw)
print("-----------------")
print("\nFrench Translation:")
for i in range(len(predictions)):
    print("---------------------------------------------")
    output_beams_per_sample = predictions[i,:,:]
    score_beams_per_sample = beam_scores[i,:,:]
    for beam, score in zip(output_beams_per_sample,score_beams_per_sample) :
        seq = list(itertools.takewhile( lambda index: index !=2, beam))
        score_indexes = np.arange(len(seq))
        beam_score = score[score_indexes].sum()
        print(" ".join( [Y_tokenizer.index_word[w] for w in seq]), " beam score: ", beam_score)

"""### Evaluate Loss"""

def eval_step(input_batch, output_batch,encoder_initial_cell_state, BATCH_SIZE):
    #initialize loss = 0
    loss = 0

    # we can do initialization in outer block
    #encoder_initial_cell_state = encoder.initialize_initial_state()
    encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
    a, h_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                    initial_state =encoder_initial_cell_state)

    #pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
    s_prev = [h_tx, c_tx]

    decoder_input = output_batch[:,:-1] # ignore <end>
    #compare logits with timestepped +1 version of decoder_input
    decoder_output = output_batch[:,1:] #ignore <start>
    decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)
    decoder_instance = tfa.seq2seq.BasicDecoder(decoderNetwork.rnn_cell, 
                                                greedy_sampler,
                                                decoderNetwork.dense_layer)
    #BasicDecoderOutput

    decoderNetwork.attention_mechanism.setup_memory(a)
    decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,
                                                                       encoder_state=s_prev,
                                                                       Dtype=tf.float32)
    outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                           sequence_length=BATCH_SIZE*[Ty-1])
    logits = outputs.rnn_output
    sample_id = outputs.sample_id
    #Calculate loss
    loss = loss_function(logits, decoder_output)
    return loss, sample_id

"""### Evaluation Loss on Entire Test Set"""

dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(len(X_test))
for (input_batch, output_batch) in dataset_test.take(-1):
    batch_size = len(input_batch)
    print(input_batch.shape)
    encoder_initial_cell_state = [tf.zeros((batch_size, rnn_units)),
                                  tf.zeros((batch_size, rnn_units))]
    loss,_ = eval_step(input_batch, output_batch, encoder_initial_cell_state, batch_size)
    loss = tf.reduce_mean(loss)
    print("Training loss {}".format(loss) )

#BasicDecoder initialization returns the <start> sequence as first_input
#Check Inference Cell output

start_index = Y_tokenizer.word_index['<start>']
start_index = tf.constant([start_index], dtype = tf.int32)
print(start_index)
start_index_emb = decoderNetwork.decoder_embedding(start_index)
print(start_index_emb.shape)
start_index_emb_avg = tf.reduce_sum(start_index_emb)
print(start_index_emb_avg.numpy())