# -*- coding:utf-8 -*-

import tensorflow as tf
from AttentionRNN import AttentionBiRNN
from utils import InputHelper
import time
import os
import numpy as np

# Parameters
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN, and dimensionality of character embedding(default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 10, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 100, 'attention layer size')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_string('train_file', 'train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model saved directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log info directory')
tf.flags.DEFINE_string('pre_trained_vec', None, 'using pre trained word embeddings, npy file format')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_integer('save_steps', 1000, 'num of train steps for saving model')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{0}={1}'.format(attr.upper(), value))


def train():
    test_data_loader = InputHelper()
    test_data_loader.load_dictionary(FLAGS.data_dir + '/dictionary')
    test_data_loader.create_batches(FLAGS.data_dir + '/' + FLAGS.test_file, 100, FLAGS.sequence_length)
    FLAGS.vocab_size = test_data_loader.vocab_size
    FLAGS.n_classes = test_data_loader.n_classes
    FLAGS.num_batches = test_data_loader.num_batches

    # Define specified Model
    model = AttentionBiRNN(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size, layer_size=FLAGS.layer_size,
                           vocab_size=FLAGS.vocab_size, attn_size=FLAGS.attn_size,
                           sequence_length=FLAGS.sequence_length,
                           n_classes=FLAGS.n_classes, grad_clip=FLAGS.grad_clip, learning_rate=FLAGS.learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            x = raw_input('请输入一个地址:\n')
            x = [data_loader.transform_raw(x, FLAGS.sequence_length)]

            labels, alphas = model.inference(sess, data_loader.labels, x)
            print labels


if __name__ == '__main__':
    main()