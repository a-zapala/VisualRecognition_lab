import os
from math import floor

import numpy as np

# !wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

FOLD_DEGREE = 4
DATA_PATH = "./cifar-10-batches-py/"
METRIC = 1


def unpickle(file):  # read data
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def reduce_data(data):
    data['imgs'] = data['imgs'][0::20]
    data['labels'] = data['labels'][0::20]
    return data


def read_test_data(directory_name):
    dict = {}
    dict_data = unpickle(os.path.join(directory_name, "test_batch"))
    dict['imgs'] = np.array(dict_data[b'data'], dtype=np.int32)
    dict['labels'] = np.array(dict_data[b'labels']).reshape(len(dict_data[b'labels']), 1)
    return dict


def get_fold(data, fold):
    train_set = {}
    test_set = {}
    size = data['labels'].size
    portion_size = floor(size / FOLD_DEGREE)
    test_set['labels'] = data['labels'][portion_size * fold:(fold + 1) * portion_size]
    test_set['imgs'] = data['imgs'][portion_size * fold:(fold + 1) * portion_size, :]
    train_set['labels'] = np.concatenate(
        (data['labels'][0:portion_size * fold], data['labels'][portion_size * (fold + 1):]))
    train_set['imgs'] = np.vstack((data['imgs'][0:portion_size * fold, :],
                                   data['imgs'][portion_size * (fold + 1):, :]))
    return test_set, train_set


def read_train_data(directory_name):
    images = np.array([])
    labels = []
    for i in range(1, 6):
        dict_data = unpickle(directory_name + "data_batch_" + str(i))
        images = np.vstack((images, dict_data[b'data'])) if images.size else np.array(
            dict_data[b'data'], dtype=np.int32)  # check which integer is better
        labels = labels + dict_data[b'labels']
    dict = {}
    dict['imgs'] = images
    dict['labels'] = np.asarray(labels).reshape(len(labels), 1)
    return dict


def join_data_set(set1, set2):
    join_data = {}
    join_data['imgs'] = np.vstack((set1['imgs'], set2['imgs']))
    join_data['labels'] = np.concatenate((set1['labels'], set2['labels']))
    return join_data


def get_prediction(data, image, metric, k):
    diff = np.array(data['imgs'] - image)
    diff = np.absolute(diff)
    diff = diff ** metric
    diff = np.sum(diff, axis=1, keepdims=True)
    arr = np.concatenate((diff, data['labels']), axis=1)
    arr = arr[arr[:, 0].argsort()]  # sorting based on first column
    arr = arr[0:k, 1]
    counts = np.bincount(arr)
    return np.argmax(counts)


def test_knn(train_set, test_set, metric, k):
    predicted = 0
    size = test_set['labels'].size
    for i in range(size):
        if get_prediction(train_set, test_set['imgs'][i], metric, k) == test_set['labels'][i]:
            predicted += 1
    return predicted / size


train_data = reduce_data(read_train_data(DATA_PATH))
test_data = reduce_data(read_test_data(DATA_PATH))

k_selected = 0
best_accuracy = 0

for k in [1, 3, 5, 7]:
    print("k-NearestNeighbors for k =", k)
    average_accuracy = 0
    for fold in range(4):
        test_set, train_set = get_fold(train_data, fold)
        accuracy = test_knn(train_set, test_set, METRIC, k)
        average_accuracy += accuracy
        print("fold =", fold, " accuracy =", accuracy)
    average_accuracy /= 4
    if best_accuracy < average_accuracy:
        best_accuracy = average_accuracy
        k_selected = k

final_accuracy = test_knn(train_data, test_data, METRIC, k_selected)
print("\nSekected k for k-NearestNeighbors =", k_selected, " accuracy =", final_accuracy)
