from sklearn.model_selection import train_test_split
import numpy as np
from random import uniform
import pdb

def load_tsv(filename):
    X = []
    Y = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        sentence_id = 1
        for line in lines[1:len(lines)+1]:
            vs = line.strip().split('\t')
            # parase_id = int(vs[0])
            sentence_id_now = int(vs[1])
            if sentence_id_now > sentence_id:
                x = vs[2]
                y = vs[3]
                sentence_id += 1
                X.append(x)
                Y.append(y)
    return X, Y
X, y = load_tsv('Data/train.tsv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_train = [int(i) for i in y_train]
y_test = [int(i) for i in y_test]
chars = list(set(' '.join(X)))
data_size, vocab_size = len(X_train), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(5, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((5, 1))  # output bias

def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs = {}, {}
    ys, ps = 0, 0
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
    ys = np.dot(Why, hs[len(inputs)-1]) + by  # unnormalized log probabilities for next chars
    ps = np.exp(ys) / np.sum(np.exp(ys))  # probabilities for next chars
    loss += np.sum(-np.log(ps))  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    dy = np.copy(ps)
    dy[targets] -= 1
    for t in reversed(range(len(inputs))):
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    # pdb.set_trace()
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

# gradient checking
def gradCheck(inputs, targets, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (s0, s1)
    print(name)
    for i in range(num_checks):
      ri = int(uniform(0, param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
      # rel_error should be on order of 1e-7 or less

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while n < len(X_train)*2:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    sentence_id = np.random.randint(0, len(X_train))
    hprev = np.zeros((hidden_size, 1))  # reset RNN memory
    inputs = [char_to_ix[ch] for ch in X_train[sentence_id][:]]
    targets = y_train[sentence_id]
    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 1000 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))  # print progress
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
    p += seq_length  # move data pointer
    n += 1  # iteration counter

def eval(inputs, targets):
    xs, hs = {}, {}
    hs[-1] = np.copy(hprev)
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
    ys = np.dot(Why, hs[len(inputs) - 1]) + by  # unnormalized log probabilities for next chars
    ps = np.exp(ys) / np.sum(np.exp(ys))  # probabilities for next chars
    y_predict = np.argmax(ps)
    return y_predict
right_num = 0
for i in range(len(X_test)):
    inputs = [char_to_ix[ch] for ch in X_test[i][:]]
    targets = y_test[i]
    y_predict = eval(inputs, targets)
    if y_predict == targets:
        right_num += 1
test_accu = right_num/len(X_test)
print('test_accu:',test_accu)

