import tensorflow as tf
import numpy as np

# Inputs: sequences of 1024-entry vectors
# Input shape is [batch_size, sequence_length, encoded_size]
#
# Outputs: 3D-tensor that can feed into the decoder
# Output shape = [batch_size, 4, 4, 4, Nh]
# where Nh is the hidden state size
def simple_lstm(input_seq_encoded, Nh=256):
    num_hidden = 4**3 * Nh
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    hidden_state, cell_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    hidden_state = tf.transpose(hidden_state, [1, 0, 2])
    outputs = hidden_state[-1]
    outputs = tf.reshape(outputs, [-1, 4, 4 ,4, Nh])
    return outputs

def test_simple_lstm():
    # Just to test if the LSTM works
    batch_size = 2
    seq_length = 5
    encoded_size = 10

    X = np.random.randint(2, size=[batch_size, seq_length, encoded_size])

    inputs = tf.placeholder(tf.float32, [None, seq_length, encoded_size])
    outputs = simple_lstm(inputs, Nh=16)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    y = sess.run(outputs, feed_dict={inputs: X})
