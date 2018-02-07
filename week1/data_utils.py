#!/usr/bin/python
# _*_coding:utf-8_*_

"""
data processing
"""
from __future__ import print_function, division
import numpy as np
import pickle
import fool
import re
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


class DataHelper(object):
    """
    数据预处理
    """
    def data_split(self, path):
        contents = []
        with open(path) as f:
            for line in f:
                contents.append(line.replace('\n', ''))

        np.random.shuffle(contents)
        with open('data/train.txt', 'w') as f:
            for line in contents:
                f.writelines(line + '\n')

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
                if token not in token_dictionary:
                    token_dictionary[token] = token_index
                    token_index += 1

        token_dictionary['</s>'] = token_index
        token_index += 1
        vocab_size = len(token_dictionary)
        n_classes = len(label_dictionary)
        print('Corpus Vocabulary:{0}, Classes:{1}'.format(vocab_size, n_classes))

        with open(save_dir + 'dictionary', 'w') as f:
            pickle.dump((token_dictionary, label_dictionary), f)
        print('creating dictionary is completed!!!')

    def load_dictionary(self, dictionary_file):
        """
        load dictionary
        :param dictionary_file:
        :return: token_dictionary, label_dictionary, vocab_size, n_classes, labels
        """
        with open(dictionary_file) as f:
            token_dictionary, label_dictionary = pickle.load(f)
            vocab_size = len(token_dictionary)
            n_classes = len(label_dictionary)

            labels = [None for _ in range(n_classes)]
            for key in label_dictionary:
                labels[label_dictionary[key]] = key
        self.token_dictionary = token_dictionary
        self.label_dictionary = label_dictionary
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.labels = labels

    def create_batches(self, train_file, batch_size, sequence_length):
        """
        create batches
        :param train_file:
        :param batch_size:
        :param sequence_length:
        :param dictionary_list:
        :return:
        """
        self.x_data = []
        self.y_data = []
        padding_index = self.vocab_size - 1
        for line in open(train_file):
            line = line.decode('utf-8').replace('\n', '')
            text, label = line.strip().split('\t')
            tokens = fool.cut(re.sub(r'\w+', ' L', text))
            seq_ids = [self.token_dictionary.get(token) for token in tokens[0]
                       if self.token_dictionary.get(token) is not None]
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

        x = [self.token_dictionary.get(token) for token in fool.cut(re.sub(r'\w+', ' L', text))[0]]
        x = x[:sequence_length]
        padding_index = self.vocab_size - 1
        for _ in range(len(x), sequence_length):
            x.append(padding_index)
        return x


if __name__ == '__main__':
    data_loader = DataHelper()
