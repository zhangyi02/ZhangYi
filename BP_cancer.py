# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

# %%
# import some data to play with
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# shuffle
idx = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idx)
X = X[idx]
y = y[idx].reshape(y.shape[0],1)

# standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

#print(X.shape, y.shape)

# split train and test
X_train, y_train = X[0:499, :], y[0:499]
X_test, y_test = X[500:568, :], y[500:568]

## Define the data set XOR
#X = np.array([
#    [1, 1, 1],
#    [1, 0, 1],
#    [0, 1, 1],
#    [0, 0, 1],
#])
#
#y = np.array([[0],
#              [1],
#              [1],
#              [0]
#             ])

# %% define sigmoid function
def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# %%
# Define a learning rate
eta = 0.01
# Define the number of epochs for learning
epochs = 5000

# Initialize the weights with random numbers
np.random.seed(100)
w01 = np.random.random((X.shape[1], 20))
b01 = np.random.random((1, 20))
w12 = np.random.random((20, 10))
b12 = np.random.random((1, 10))
w23 = np.random.random((10,1))
b23 = np.random.random((1,1))
#w01 = np.zeros((len(X[0]), 5))
#w12 = np.zeros((5,1))

# Creat a list to store a_o_error
error_list = []
error_list_sum = []

X = X_train
y = y_train
# Start feeding forward and backpropagate *epochs* times.
for epoch in range(epochs):
    # Feed forward
    z_h1 = np.dot(X, w01) + b01
    a_h1 = sigmoid(z_h1)
    
    z_h2 = np.dot(a_h1, w12) + b12
    a_h2 = sigmoid(z_h2)

    z_o = np.dot(a_h2, w23) + b23
    a_o = sigmoid(z_o)

    # Calculate the error
    a_o_error = ((1 / 2) * (np.power((a_o - y), 2)))
    error_list.append(a_o_error)
    error_list_sum.append(sum(a_o_error))
    
    # Backpropagation
    ## Output layer
    da_o = a_o - y
    dz_o = da_o * sigmoid(a_o,derive=True)
    dw23 = (a_h2.T).dot(dz_o)
    db23 = dz_o

    ## Hidden layer2
    da_h2 = dz_o.dot(w23.T)
    dz_h2 = da_h2 * sigmoid(a_h2,derive=True)
    dw12 = (a_h1.T).dot(dz_h2)
    db12 = dz_h2
    
    ## Hidden layer1
    da_h1 = dz_h2.dot(w12.T)
    dz_h1 = da_h1 * sigmoid(a_h1,derive=True)
    dw01 = (X.T).dot(dz_h1)
    db01 = dz_h1

    w01 = w01 - eta * dw01
    w12 = w12 - eta * dw12
    w23 = w23 - eta * dw23
    b01 += -eta * b01
    b12 += -eta * b12
    b23 += -eta * b23
    print(a_o_error)
#print(error_list[0])
#print(error_list[-1])
# plt.figure()
# axis_x = np.linspace(0,epochs,epochs)
# plt.plot(axis_x,error_list_sum)
# plt.show()
# %%
wrong_num = 0
error_list_test = []
for i in range(X_test.shape[0]):
    x = X_test[i,:]
    y = y_test[i]
    z_h1_test = np.dot(x, w01) + b01
    a_h1_test = sigmoid(z_h1_test)
    
    z_h2_test = np.dot(a_h1_test, w12) + b12
    a_h2_test = sigmoid(z_h2_test)
    
    z_o_test = np.dot(a_h2_test, w23) +b23
    a_o_test = sigmoid(z_o_test)
    
    # Calculate the error
    if (a_o_test > 0.5 and y > 0.5) or\
            (a_o_test <= 0.5 and y <= 0.5):
        wrong_num += 0
    else:
        wrong_num += 1
    a_o_error_test = ((1 / 2) * (np.power((a_o_test - y), 2)))
    error_list_test.append(a_o_error_test)
error_test = sum(error_list_test)
print(wrong_num)

