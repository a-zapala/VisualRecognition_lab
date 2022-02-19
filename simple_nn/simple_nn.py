# Lab 3
# Implement in Python + Numpy
# a Neural Network ( sigmoid( W0 * sigmoid( W1 * x ) ) )
# + Gradient Descent.
# Use 1/2 L2 as loss function. Using MNIST dataset

# This solution achieved 94% accuracy after 30 epochs

import pickle

import numpy as np

# https://academictorrents.com/details/323a0048d87ca79b68f12a6350a57776b6a3b7fb <- data from, standard mnist
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
training_data, test_data = data[0], data[2]


np.random.seed( 1000 )

n_input, n_hidden, n_output = training_data[0][0].size ,100, 10
biases = [ np.random.randn(n_hidden, 1), np.random.randn(n_output, 1) ]
weights = [ np.random.randn(n_hidden, n_input), np.random.randn(n_output, n_hidden) ]

n_epochs, lr, mini_batches = 500, 3.0, 500

def sigmoid(z, deriv = False):
    if(deriv):
        return sigmoid(z) * (1 - sigmoid(z))
    else:
        return 1./ (1 + np.exp(-1 * z))


def forward(x):
    wxb0 = np.dot(weights[0],x) + biases[0]
    hidden = sigmoid(wxb0)
    wxb1 = np.dot(weights[1],hidden) + biases[1]
    output = sigmoid(wxb1)
    return wxb0, hidden, wxb1, output


def backprop(x, y):
    nabla_b = [ np.zeros(biases[0].shape), np.zeros(biases[1].shape) ]
    nabla_w = [ np.zeros(weights[0].shape), np.zeros(weights[1].shape) ]

    # forward pass
    wxb0, hidden, wxb1, output = forward( x )


    # backward pass
    nabla_b[1] = 2 * (output - y) * sigmoid(wxb1, True)
    nabla_w[1] = np.dot(nabla_b[1], hidden.T)
    nabla_b[0] = np.dot(nabla_b[1].T, weights[1]).T * sigmoid(wxb0, True)
    nabla_w[0] = np.dot(nabla_b[0], x.T)
    return nabla_w, nabla_b


for ep in range(n_epochs):
    number_in_one_batch = len(training_data) / mini_batches
    # train
    nabla_w = [np.zeros(weights[0].shape), np.zeros(weights[1].shape)]
    nabla_b = [np.zeros(biases[0].shape), np.zeros(biases[1].shape)]
    i = 0
    for x, y in training_data:
        nabla_wi, nabla_bi = backprop(x, y)
        nabla_w = [nw + nwi for nw, nwi in zip(nabla_w, nabla_wi)]
        nabla_b = [nb + nbi for nb, nbi in zip(nabla_b, nabla_bi)]
        i+=1
        if i == number_in_one_batch:
            i = 0
            weights = [w - lr * nw/ number_in_one_batch for w, nw in zip(weights, nabla_w)]
            biases = [w - lr * nw/ number_in_one_batch for w, nw in zip(biases, nabla_b)]
            nabla_w = [np.zeros(weights[0].shape), np.zeros(weights[1].shape)]
            nabla_b = [np.zeros(biases[0].shape), np.zeros(biases[1].shape)]

    # evaluate
    s = 0
    for x, y in test_data:
        _, _, _, output = forward( x )
        s += int(np.argmax(output) == y)
    print("Epoch {} : {} / {}".format( ep, s, len(test_data)))
