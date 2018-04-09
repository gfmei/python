#!/usr/bin/env python
# -*-encoding=utf-8-*-

import tensorflow as tf
from tensorflow.contrib import rnn


class BiMPM(object):
    def __init__(self, config, is_training=True):
        num_step1 = config.num_step1
        num_step2 = config.num_step2
        embedding_size = config.embedding_size
        vocab_size = config.vocab_size
        l2_reg_lambda = config.l2_reg_lambda
        self.class_num = config.class_num
        self.input_s1 = tf.placeholder(tf.int32, [None, num_step1])
        self.input_s2 = tf.placeholder(tf.int32, [None, num_step2])
        self.target = tf.placeholder(tf.int32, [None, self.class_num])
        self.epsilon = config.epsilon
        self.match_dim = config.match_dim
        hl1_embedding_size = config.hl1_embedding_size
        hl2_embedding_size = config.hl2_embedding_size
        self.dropout_rate = config.dropout_rate
        with tf.name_scope('embedding_layer'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32, name="W")
            self.embedded_s1 = tf.nn.embedding_lookup(self.W, self.input_s1)
            self.embedded_s2 = tf.nn.embedding_lookup(self.W, self.input_s2)
        ((cr_s1_fw_outputs, cr_s1_bw_outputs), _) = self.bi_rnn(hl1_embedding_size, self.embedded_s1, 'cr_s1')
        ((cr_s2_fw_outputs, cr_s2_bw_outputs), _) = self.bi_rnn(hl1_embedding_size, self.embedded_s2, 'cr_s2')
        self.cr_s1_outputs = tf.concat((cr_s1_fw_outputs, cr_s1_bw_outputs), 2)
        self.cr_s2_outputs = tf.concat((cr_s2_fw_outputs, cr_s2_bw_outputs), 2)
        with tf.name_scope('matching_operation'):
            self.mW = tf.Variable(tf.truncated_normal([self.match_dim, hl1_embedding_size*2], stddev=0.1), name='mW')
            cosine_matrix_s1 = self.cosine_matrix(self.cr_s1_outputs, self.cr_s2_outputs)
            cosine_matrix_s2 = self.cosine_matrix(self.cr_s2_outputs, self.cr_s1_outputs)
            self.match_s1 = self.attentive_matching(self.cr_s1_outputs, self.cr_s2_outputs, cosine_matrix_s1, self.mW)
            self.match_s2 = self.attentive_matching(self.cr_s2_outputs, self.cr_s1_outputs, cosine_matrix_s2, self.mW)
        # outputs: A tuple (output_fw, output_bw)
        (_, (cr_s1_fw_output_states, cr_s1_bw_output_states)) = self.bi_rnn(hl2_embedding_size, self.match_s1, 'rnn_s1')
        (_, (cr_s2_fw_output_states, cr_s2_bw_output_states)) = self.bi_rnn(hl2_embedding_size, self.match_s2, 'rnn_s2')
        with tf.name_scope('aggregation_layer'):
            self.ag_vector = tf.concat((tf.concat((cr_s1_fw_output_states.c, cr_s1_bw_output_states.c), 1),
                                        tf.concat((cr_s2_fw_output_states.c, cr_s2_bw_output_states.c), 1)), 1)

        # Dropout
        flag = True
        if not is_training:
            flag = False
        with tf.name_scope("dropout"):
            drop = self.dropout_layer(self.ag_vector, self.dropout_rate, is_training=flag)

        # Fully connected layer
        with tf.name_scope("full_network"):
            fW = tf.Variable(tf.truncated_normal([hl2_embedding_size * 4, self.class_num], stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[self.class_num]), name="b")
            self.y_hat = tf.nn.xw_plus_b(drop, fW, b, name="y_hat")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(fW)
        l2_loss += tf.nn.l2_loss(b)

        # Cross-entropy loss and optimizer initialization
        with tf.name_scope("output"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_hat, labels=self.target),
                                       name='loss')
            self.loss += l2_reg_lambda * l2_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
            self.predictions = tf.argmax(self.y_hat, 1, name="predictions")
            self.prob = tf.nn.softmax(self.y_hat, dim=-1, name='probability')

        # Accuracy metric
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.target, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def bi_rnn(self, embedding_size, inputs, scope, forget_bias=1.0):
        with tf.name_scope('BiRNN_' + scope), tf.variable_scope('BiRNN_' + scope, dtype=tf.float32):
            cell_fw = rnn.BasicLSTMCell(embedding_size, forget_bias=forget_bias)
            cell_bw = rnn.BasicLSTMCell(embedding_size, forget_bias=forget_bias)
            # [batch_size, max_time, cell_fw.output_size]
            (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                                       time_major=False, dtype=tf.float32)
            return outputs, output_states

    def cosine_similarity(self, y1, y2, cosine_norm=True, eps=1e-6):
        """Compute cosine similarity.
        # Arguments:
            y1: (batch_size, len1, embedding_size)
            y2: (batch_size, len2, embedding_size)
        """
        cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
        if not cosine_norm:
            return tf.tanh(cosine_numerator)
        y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
        y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
        return cosine_numerator / y1_norm / y2_norm

    def calculate_attention(self, s1, s2):
        # s1_shape = tf.shape(s1)  # [batch_size, len_1, rnn_dim]
        s1_expand = tf.expand_dims(s1, 1)
        s2_expand = tf.expand_dims(s2, 2)
        alpha = self.cosine_similarity(s1_expand, s2_expand)
        return alpha

    def cosine_matrix(self, x1, x2):
        """Cosine similarity matrix.

        Calculate the cosine similarities between each forward (or backward)
        contextual embedding h_i_p and every forward (or backward)
        contextual embeddings of the other sentence

        # Arguments
            x1: (batch_size, x1_timesteps, embedding_size)
            x2: (batch_size, x2_timesteps, embedding_size)

        # Output shape
            (batch_size, x1_timesteps, x2_timesteps)
        """
        # expand h1 shape to (batch_size, x1_timesteps, 1, embedding_size)
        x1 = tf.expand_dims(x1, axis=2)
        # expand x2 shape to (batch_size, 1, x2_timesteps, embedding_size)
        x2 = tf.expand_dims(x2, axis=1)
        # cosine matrix (batch_size, h1_timesteps, h2_timesteps)
        cos_matrix = self.cosine_similarity(x1, x2)
        return cos_matrix

    def mean_attentive_vectors(self, x2, cosine_matrix):
        """Mean attentive vectors.

        Calculate mean attentive vector for the entire sentence by weighted
        summing all the contextual embeddings of the entire sentence

        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)
        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps, x2_timesteps, 1)
        expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1)
        # (batch_size, 1, x2_timesteps, embedding_size)
        x2 = tf.expand_dims(x2, axis=1)
        # (batch_size, x1_timesteps, embedding_size)
        weighted_sum = tf.reduce_sum(expanded_cosine_matrix * x2, axis=2)
        # (batch_size, x1_timesteps, 1)
        sum_cosine = tf.expand_dims(tf.maximum(tf.reduce_sum(cosine_matrix, axis=-1), self.epsilon), axis=-1)
        # (batch_size, x1_timesteps, embedding_size)
        attentive_vector = weighted_sum / sum_cosine
        return attentive_vector

    def time_distributed_multiply(self, x, w):
        """Element-wise multiply vector and weights.

        # Arguments
            x: sequence of hidden states, (batch_size, ?, embedding_size)
            w: weights of one matching strategy of one direction,
               (match_dim, embedding_size)

        # Output shape
            (?, match_dim, embedding_size)
        """
        # dimension of vector
        shape = tf.shape(x)
        embedding_size = shape[-1]

        # collapse time dimension and batch dimension together
        x = tf.reshape(x, [-1, embedding_size])
        # reshape to (?, 1, embedding_size)
        x = tf.expand_dims(x, axis=1)
        # reshape weights to (1, match_dim, embedding_size)
        w = tf.expand_dims(w, axis=0)
        # element-wise multiply
        x = x * w
        # reshape to original shape
        # if n_dim == 3:
        x = tf.reshape(x, (-1, shape[1], self.match_dim, embedding_size))
        #     # x.set_shape([None, None, None, embedding_size])
        # elif n_dim == 2:
        #     x = tf.reshape(x, tf.stack([-1, self.match_dim, embedding_size]))
        #     x.set_shape([None, None, embedding_size])
        return x

    def attentive_matching(self, h1, h2, cosine_matrix, w):
        """Attentive matching operation.

        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            cosine_matrix: weights of hidden state h2,
                          (batch_size, h1_timesteps, h2_timesteps)
            w: weights of one direction, (mp_dim, embedding_size)

        # Output shape
            (batch_size, h1_timesteps, match_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self.time_distributed_multiply(h1, w)
        # attentive vector (batch_size, h1_timesteps, embedding_szie)
        attentive_vec = self.mean_attentive_vectors(h2, cosine_matrix)
        # attentive_vec * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        attentive_vec = self.time_distributed_multiply(attentive_vec, w)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self.cosine_similarity(h1, attentive_vec)
        return matching

    def dropout_layer(self, input_reps, dropout_rate, is_training=True):
        if is_training:
            output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
        else:
            output_repr = input_reps
        return output_repr
