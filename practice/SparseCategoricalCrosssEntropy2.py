import tensorflow as tf
tf.enable_eager_execution()
cce = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False)
#By default, we assume that y_pred encodes a probability distribution.
# i.e. the logits have been passed through a softmax layer
sparse_input = tf.convert_to_tensor([0,1,2], dtype=tf.int32)
logits = tf.convert_to_tensor([[0.9, .05, .05], [.5, .89, .6], [.05, .01, .94]],dtype=tf.float32)
loss = cce(y_true=sparse_input, y_pred=logits
)
print('Loss: ', loss.numpy())  # Loss: 0.3239
# when from_logits = True   # Loss:  0.69814444
