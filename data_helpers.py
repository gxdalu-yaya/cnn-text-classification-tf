import numpy as np
import re
import itertools
from collections import Counter


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def load_data_and_labels2(data_file, num_classes):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(data_file, "r").readlines())
    labels = list()
    x_texts = list()
    for line in examples:
        datas = line.strip().split("\t")
        if len(datas) != 2:
            continue
        label = int(datas[0])
        question = datas[1]
        # Split by words
        word_list = [word for word in question]
        labels.append(label)
        x_texts.append(" ".join(word_list))
    #标签编码为one-hot编码
    onehot_labels = get_one_hot(labels, num_classes)
    return [x_texts, onehot_labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
