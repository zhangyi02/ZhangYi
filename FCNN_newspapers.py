# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

#  import data
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

num_train = len(newsgroups_train.data)
num_test = len(newsgroups_test.data)

vectorizer = TfidfVectorizer(max_features=500)

X = vectorizer.fit_transform(newsgroups_train.data + newsgroups_test.data)
# y = vectorizer.fit_transform( newsgroups_train.target + newsgroups_test.target )
X = X.toarray()
mean = X.mean(axis=0)
std = X.std(axis=0)
X = np.array((X - mean)) / std
X_train = X[0:num_train, :]
X_test = X[num_train:num_train + num_test, :]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    w1, b1= model['w1'], model['b1']
    w2, b2= model['w2'], model['b2']
    w3, b3= model['w3'], model['b3']

    #
    #    # Forward propagation to calculate our predictions
    #    z1 = X.dot(W1) + b1
    #    a1 = np.tanh(z1)
    #    z2 = a1.dot(W2) + b2
    #    exp_scores = np.exp(z2)
    #    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    probs = predict(model, X, y, return_probs=True)

    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    # Add regulatization term to loss (optional)
    data_loss += reg_lambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)))
    return 1. / num_examples * data_loss


# Helper function to predict an output (0 or 1)
def predict(model, X, y, return_probs=False):
    w1, b1 = model['w1'], model['b1']
    w2, b2 = model['w2'], model['b2']
    w3, b3 = model['w3'], model['b3']
    # Forward propagation
    z1 = X.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(w3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    if return_probs:
        return probs
    else:
        return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations

def build_model(nn_hdim, X, y, epochs, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(100)
    w1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    w2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_hdim))
    w3 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b3 = np.zeros((1, nn_output_dim))
    # This is what we return at the end
    model = {}
    train_loss = []
    # Gradient descent. For each batch...
    for i in range(0, epochs):

        #   Forward propagation
        z1 = X.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        a2 = np.tanh(z2)
        z3 = a2.dot(w3) + b3
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        #   Backpropagation
        delta4 = probs
        delta4[range(len(X)), y] -= 1
        # This two line is calculating p - y // and this is dz
        dw3 = (a2.T).dot(delta4) / len(X)
        db3 = np.sum(delta4, axis=0, keepdims=True) / len(X)

        delta3 = delta4.dot(w3.T) * (1 - np.power(a2, 2))
        dw2 = (a1.T).dot(delta3) / len(X)
        db2 = np.sum(delta3, axis=0) / len(X)

        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dw1 = (X.T).dot(delta2) / len(X)
        db1 = np.sum(delta2, axis=0) / len(X)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dw3 += reg_lambda * w3
        dw2 += reg_lambda * w2
        dw1 += reg_lambda * w1

        # Gradient descent parameter update
        w1 += -epsilon * dw1
        b1 += -epsilon * db1
        w2 += -epsilon * dw2
        b2 += -epsilon * db2
        w3 += -epsilon * dw3
        b3 += -epsilon * db3

        # Assign new parameters to the model
        model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 100 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))
        train_loss.append(calculate_loss(model, X, y))

    return model, train_loss


# define parameters
num_examples = len(X_train)  # training set size
nn_input_dim = X.shape[1]  # input layer dimensionality
nn_output_dim = 4  # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.1  # learning rate for gradient descent
reg_lambda = 0  # regularization strength

epochs = 1000
model, train_loss = build_model(100, X_train, Y_train, epochs, print_loss=True)
plt.figure()
axis_x = np.linspace(0, epochs, epochs)
plt.plot(axis_x, train_loss)
plt.show()


def eval(model, X, y):
    wrong_num = 0
    y_predict = predict(model, X, y)
    for i in range(len(y)):
        if y_predict[i] == y[i]:
            wrong_num += 0
        else:
            wrong_num += 1
    return wrong_num


wrong_num = eval(model, X_test, Y_test)
print(1 - wrong_num / len(Y_test))

