#!/usr/bin/python
# _*_coding:utf-8_*_

"""
用于文本分类任务
train_file为已经分好词的文本 如 'token1 token2 ... \t label'
token之间使用空格分开, 与label使用\t隔开
"""

from __future__ import print_function, division
import numpy as np
import pickle
import fool
import re
import chinese

import sys

reload(sys)
sys.setdefaultencoding("utf-8")


class InputHelper(object):
    """
    输入数据的预处理
    """
    def __init__(self, path):
        with open(path) as f:
            # path = 'data/stop_words.pkl'
            self.stop_words = pickle.load(f)

    def create_dictionary(self, train_file, save_dir):
        """
        从原始文本文件中创建字典
        :param train_file: 原始训练文件文档
        :param save_dir: 词典保存路径
        :return: token_dictionary, label_dictionary, labels, vocab_size, n_classes
        """
        token_dictionary = {}
        token_index = 0
        label_dictionary = {}
        label_index = 0
        labels = []

        for line in open(train_file):
            line = line.decode('utf-8').replace('\n', '')
            text, label = line.strip().split('\t')
            tokens = fool.cut(re.sub(r'\w+', ' L', text))
            # print(tokens)
            if label not in label_dictionary:
                label_dictionary[label] = label_index
                labels.append(label)
                label_index += 1

            for token in tokens[0]:
                if token not in token_dictionary and not chinese.is_other_all(token) and token not in self.stop_words:
                    token_dictionary[token] = token_index
                    token_index += 1

        token_dictionary['</s>'] = token_index
        token_index += 1
        self.vocab_size = len(token_dictionary)
        self.n_classes = len(label_dictionary)
        print('Corpus Vocabulary:{0}, Classes:{1}'.format(self.vocab_size, self.n_classes))

        with open(save_dir + 'dictionary', 'w') as f:
            pickle.dump((token_dictionary, label_dictionary), f)

        self.token_dictionary = token_dictionary
        self.label_dictionary = label_dictionary
        self.labels = labels

    def load_dictionary(self, dictionary_file):

        with open(dictionary_file) as f:
            self.token_dictionary, self.label_dictionary = pickle.load(f)
            self.vocab_size = len(self.token_dictionary)
            self.n_classes = len(self.label_dictionary)
            self.labels = [None for _ in range(self.n_classes)]

            for key in self.label_dictionary:
                self.labels[self.label_dictionary[key]] = key

    def create_batches(self, train_file, batch_size, sequence_length):

        self.x_data = []
        self.y_data = []
        padding_index = self.vocab_size - 1
        for line in open(train_file):
            line = line.decode('utf-8').replace('\n', '')
            text, label = line.strip().split('\t')
            tokens = fool.cut(re.sub(r'\w+', ' L', text))
            seq_ids = [self.token_dictionary.get(token) for token in tokens[0] if token not in self.stop_words and
                       self.token_dictionary.get(token) is not None and not chinese.is_other_all(token)]
            seq_ids = seq_ids[:sequence_length]
            for _ in range(len(seq_ids), sequence_length):
                seq_ids.append(padding_index)

            self.x_data.append(seq_ids)
            self.y_data.append(self.label_dictionary.get(label))

        self.num_batches = int(len(self.x_data) / batch_size)
        self.x_data = self.x_data[:self.num_batches * batch_size]
        self.y_data = self.y_data[:self.num_batches * batch_size]

        self.x_data = np.array(self.x_data, dtype=int)
        self.y_data = np.array(self.y_data, dtype=int)
        self.x_batches = np.split(self.x_data.reshape(batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(self.y_data.reshape(batch_size, -1), self.num_batches, 1)
        self.pointer = 0

    def label_one_hot(self, label_id):

        y = [0] * self.n_classes
        y[int(label_id)] = 1.0
        return np.array(y)

    def next_batch(self):
        index = self.batch_index[self.pointer]
        self.pointer += 1
        x_batch, y_batch = self.x_batches[index], self.y_batches[index]
        y_batch = [self.label_one_hot(y) for y in y_batch]
        return x_batch, y_batch

    def reset_batch(self):
        self.batch_index = np.random.permutation(self.num_batches)
        self.pointer = 0

    def transform_raw(self, text, sequence_length):

        if not isinstance(text, unicode):
            text = text.decode('utf-8')
        tokens = fool.cut(re.sub(r'\w+', ' L', text))[0]
        x = [self.token_dictionary.get(token) for token in tokens if not chinese.is_other_all(token)
             and token not in self.stop_words]
        x = x[:sequence_length]
        padding_index = self.vocab_size - 1
        for _ in range(len(x), sequence_length):
            x.append(padding_index)
        self.words = [token for token in tokens if not chinese.is_other_all(token)]
        return x


if __name__ == '__main__':
    data_loader = InputHelper('data/stop_words.pkl')