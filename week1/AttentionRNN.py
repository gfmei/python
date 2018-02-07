# -*- coding:utf-8 -*-

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


class AttentionBiRNN(object):
    """
    用于文本分类的双向RNN
    """
    def __init__(self, embedding_size, rnn_size, layer_size, vocab_size, attn_size, sequence_length, n_classes,
                 grad_clip, learning_rate):
        """
        the implementation of Bi-LSTMCell
        :param embedding_size: word embedding dimension
        :param rnn_size: hidden state dimension
        :param layer_size: number of rnn layers
        :param vocab_size: vocabulary size
        :param attn_size: attention layer dimension
        :param sequence_length: max sequence length
        :param n_classes: number of target labels
        :param grad_clip: gradient clipping threshold
        :param learning_rate: initial learning rate
        """
        self.input_data = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_data')
        self.targets = tf.placeholder(tf.float32, shape=[None, n_classes], name='targets')
        self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')

        # define the forward RNN Cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            print(tf.get_variable_scope().name)
            lstm_fw_rnn_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_fw_rnn_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_rnn_list),
                                                          output_keep_prob=self.output_keep_prob)

        # define backward RNN Cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            print(tf.get_variable_scope().name)
            lstm_bw_rnn_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_bw_rnn_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_rnn_list),
                                                          output_keep_prob=self.output_keep_prob)

        with tf.name_scope('embedding'), tf.variable_scope('embedding'):
            embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1), name='embedding')
            inputs = tf.nn.embedding_lookup(embedding, self.input_data, name='look_up')

        """
        self.input_data shape: (batch_size , sequence_length)
        inputs shape: (batch_size , sequence_length , rnn_size)
        bi-direction rnn 的inputs shape 要求是(sequence_length, batch_size, rnn_size)
        因此这里需要对inputs做一些变换, 经过transpose的转换已经将shape变为(sequence_length, batch_size, rnn_size)
        只是双向rnn接受的输入必须是一个list,因此还需要后续两个步骤的变换
        """
        # batch_size 与 sequence_length维互换
        inputs = tf.transpose(inputs, [1, 0, 2])
        # 转换成(batch_size * sequence_length, rnn_size)
        inputs = tf.reshape(inputs, [-1, rnn_size])
        # 转换成list,里面的每个元素是(batch_size, rnn_size)
        inputs = tf.split(inputs, sequence_length, 0)
        print(tf.convert_to_tensor(inputs).shape)

        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_rnn_m, lstm_bw_rnn_m, inputs,
                                                                         dtype=tf.float32)

        # define attention layer
        attention_size = attn_size
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            self.final_output, self.alphas = self.attention(tf.convert_to_tensor(self.outputs), attention_size,
                                                            time_major=True, return_alphas=True)

        # outputs shape: (sequence_length, batch_size, 2*rnn_size)
        fc_w = tf.Variable(tf.truncated_normal([2 * rnn_size, n_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([n_classes]), name='fc_b')

        # self.final_output = outputs[-1]

        # 用于分类任务, outputs取最终一个时刻的输出
        self.logits = tf.matmul(self.final_output, fc_w) + fc_b
        self.prob = tf.nn.softmax(self.logits)

        self.cost = tf.losses.softmax_cross_entropy(self.targets, self.logits)
        # self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=tf.squeeze(self.logits), labels=self.targets)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.targets, axis=1), tf.argmax(self.prob, axis=1)), tf.float32))

    # attention function
    def attention(self, inputs, attention_size, time_major=False, return_alphas=False):
        """
         Attention mechanism layer which reduces Bi-RNN outputs with Attention vector.
         Args:
             inputs: The Attention inputs.
             attention_size: the dim of weights vector
             time_major:
             return_alphas: the weight vector
         Returns:
             The Attention output `Tensor`.
             In case of RNN, this will be a `Tensor` shaped:
                 `[batch_size, cell.output_size]`.
             In case of Bidirectional RNN, this will be a `Tensor` shaped:
                 `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
         """

        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape
        alphas = tf.nn.softmax(vu)  # (B,T) shape also

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    # prediction
    def inference(self, sess, labels, inputs):
        prob, alphas, output = sess.run([self.prob, self.alphas, self.outputs],
                                        feed_dict={self.input_data: inputs, self.output_keep_prob: 1.0})
        ret = np.argmax(prob, 1)
        print(len(output))
        ret = [labels[i] for i in ret]
        return ret, alphas


if __name__ == '__main__':
    model = AttentionBiRNN(128, 128, 2, 100, 256, 50, 30, 5, 0.001)