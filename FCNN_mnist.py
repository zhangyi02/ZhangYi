#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:41:07 2018

@author: zhangyi
"""
# Import packages
import numpy as np
import sys
import mnist

# Import data
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

n_train, w, h = train_images.shape
X_train = train_images.reshape( (n_train, w*h) )
Y_train = train_labels

n_test, w, h = test_images.shape
X_test = test_images.reshape( (n_test, w*h) )
Y_test = test_labels

#print(X_train.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)

#  Define sigmoid function
def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_scores = np.exp(x)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

# Define predict function
def predict(model, X, y):
    w1, b1= model['w1'], model['b1']
    w2, b2 = model['w2'], model['b2']
    w3, b3 = model['w3'], model['b3']
    
#    first layer - calculate input to a1
    z1 = X.dot(w1) + b1
    a1 = sigmoid(z1)
#    second layer - calculate a1 to a2
    z2 = a1.dot(w2) + b2
    a2 = sigmoid(z2)
#    third layer - calculate a2 to a3
    z3 = a2.dot(w3) + b3
    a3 = softmax(z3)
#   return the output
    return np.argmax(a3, axis=1)

# Define cost function
def cost(X, y):
#    Calculating the loss
    probs = a3
    corect_logprobs = -np.log(probs[range(len(X)), y])
    data_loss = np.sum(corect_logprobs)
    
#    Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2))
    + np.sum(np.square(w3)))
    return 1./len(X) * data_loss

# Define eval function
def eval(model, X, y):
    right_num = 0
    y_predict = predict(model, X, y)
    for i in range(len(y)):
        if y_predict[i] == y[i]:
            right_num += 1
    accu_rate = 1 - right_num/len(y)
    return accu_rate

# Define learning function
def train(model, X, y, lr, epochs):
    w1, b1= model['w1'], model['b1']
    w2, b2 = model['w2'], model['b2']
    w3, b3 = model['w3'], model['b3']
    for epoch in range(epochs):     
#    Forward propagation
#    First layer - calculate input to a1
        z1 = X.dot(w1) + b1
        a1 = sigmoid(z1)
#    Second layer - calculate a1 to a2
        z2 = a1.dot(w2) + b2
        a2 = sigmoid(z2)
#    Third layer - calculate a2 to a3
        z3 = a2.dot(w3) + b3
        a3 = softmax(z3)
    
#    Calculating the cost
#        a3_error = cost(X, y)
        probs = a3
        corect_logprobs = -np.log(probs[range(len(X)), y])
        data_loss = np.sum(corect_logprobs)
    
#    Add regulatization term to loss (optional)
#        data_loss += reg_lambda/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2))
#        + np.sum(np.square(w3)))
        cost_list.append(1./len(X) * data_loss)

#    Backward propagation
        dz3 = a3
        dz3[range(len(X)), y] -= 1
        dw3 = (a2.T).dot(dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3.dot(w3.T)
        dz2 = da2 * sigmoid(a2, derive=True)
        dw2 = (a1.T).dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
    
        da1 = dz2.dot(w2.T)
        dz1 = da1 * sigmoid(a1, derive=True)
        dw1 = (X.T).dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
    
#    Gradient descent parameter update
        w1 += -lr * dw1
        b1 += -lr * db1
        w2 += -lr * dw2
        b2 += -lr * db2
        w3 += -lr * dw3
        b3 += -lr * db3

    model = { 'w1': w1, 'b1': b1, 
             'w2': w2, 'b2': b2,
             'w3': w3, 'b3': b3}
    return model,cost_list
    
    
# Define learning rate and the number of epochs for learning
lr = 0.01
epochs = 1000

# Define nn-num
nn_num = [20,10]

# Initialize the weights with random numbers
np.random.seed(100)
w1 = np.random.randn(X_train.shape[1], nn_num[0])/np.sqrt(X_train.shape[1])
b1 = np.zeros((1, nn_num[0]))
w2 = np.random.randn(nn_num[0], nn_num[1])/np.sqrt(nn_num[0])
b2 = np.zeros((1, nn_num[1]))
w3 = np.random.randn(nn_num[1],10)/np.sqrt(nn_num[1])
b3 = np.zeros((1, 10))
cost_list = []
model = { 'w1': w1, 'b1': b1, 
         'w2': w2, 'b2': b2,
         'w3': w3, 'b3': b3}

batch_size = 20
batch_num = len(X_train)/batch_size
for i in range(int(batch_num)):
    X = X_train[0+i*batch_size:batch_size+i*batch_size]
    Y = Y_train[0+i*batch_size:batch_size+i*batch_size]
    model, cost_list = train(model, X, Y, lr, epochs)

plt.figure()
axis_x = np.linspace(0, epochs, epochs)
plt.plot(axis_x,train_loss)
plt.show()

accu_rate = eval(model, X_test, Y_test)
print('Accu_rate:',accu_rate)




    
