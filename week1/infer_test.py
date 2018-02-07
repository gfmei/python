# -*- coding:utf-8 -*-

import tensorflow as tf
from utils import InputHelper
from AttentionRNN import AttentionBiRNN

# Parameters
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN, and dimensionality of character embedding(default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 1, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 1, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 10, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 30, 'attention layer size')
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


def main():
    data_loader = InputHelper('data/stop_words.pkl')
    data_loader.load_dictionary(FLAGS.data_dir + '/dictionary')
    FLAGS.vocab_size = data_loader.vocab_size
    FLAGS.n_classes = data_loader.n_classes

    # Define specified Model
    model = AttentionBiRNN(FLAGS.embedding_size, FLAGS.rnn_size, FLAGS.layer_size, FLAGS.vocab_size, FLAGS.attn_size,
                           FLAGS.sequence_length, FLAGS.n_classes, FLAGS.grad_clip, FLAGS.learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # while True:
        # x = raw_input('请输入一个地址:\n')

        with open('./data/test.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                text = line.split('\t')
                l = text[0]
                try:
                    x = [data_loader.transform_raw(l, FLAGS.sequence_length)]
                    words = data_loader.words
                    labels, alphas, _ = model.inference(sess, data_loader.labels, x)
                    print(labels, text[1].replace('\n', ''), len(words))
                    words_weights = []
                    for word, alpha in zip(words, alphas[0] / alphas[0][0:len(words)].max()):
                        words_weights.append(word + ':' + str(alpha))
                    print(str(words_weights).decode('unicode-escape'))
                except:
                    print(l)


# with open("./save/visualization.html", "w") as html_file:
# 	html_file.write('<meta charset="UTF-8">\n')
# 	count = 0
# 	for word, alpha in zip(words, alphas[0] / alphas[0][0:len(words)].max()):
# 		print(word)
# 		if count < len(words):
# 			html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))


if __name__ == '__main__':
    main()