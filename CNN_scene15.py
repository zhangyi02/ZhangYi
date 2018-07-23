# Load data
import cv2
import os
import numpy as np
import random
import tensorflow as tf
import pdb
from sklearn.preprocessing import scale


def pre_processing():
    img_name_list = {}
    img_name_train_list = {}
    img_name_test_list = {}
    img_name_train_total = []
    img_name_test_total = []
    label_train_total = []
    label_test_total = []
    img_class = ['bedroom', 'CALsuburb', 'industrial', 'kitchen', 'livingroom',
                 'MITcoast', 'MITforest', 'MIThighway', 'MITinsidecity', 'MITmountain',
                 'MITopencountry', 'MITstreet', 'MITtallbuilding', 'PARoffice', 'store']
                 for i in range(len(img_class)):
                     path_class = 'scene_categories/' + img_class[i]
                     img_name_list[i] = os.listdir(path_class)  # 列出文件夹下所有的目录与文件
                     
                     img_name_train_list[i] = random.sample(img_name_list[i], 100)
                     img_name_test_list[i] = list(set(img_name_list[i]) - set(img_name_train_list[i]))
                     #   至此读出了所有文件路径存储于img_train_list, img_test_list两个字典中，key为label的index，对应的为该类下所有图片
                     
                     img_name_train_total += img_name_train_list[i]
                     label_train_total += [i] * len(img_name_train_list[i])
                     img_name_test_total += img_name_test_list[i]
                         label_test_total += [i] * len(img_name_test_list[i])
                     #   将各个label训练、测试数据分别合并为一个list

train_index = np.arange(len(img_name_train_total))
np.random.shuffle(train_index)
img_name_train_total_shuffle = [img_name_train_total[index] for index in train_index]
label_train_total_shuffle = [label_train_total[index] for index in train_index]
# shuffle 数据

batch_size = 50
    batch_nums = len(img_name_train_total_shuffle) // batch_size
    img_name_batch = {}
    label_batch = {}
    for batch_num in range(batch_nums):
        img_name_batch[batch_num] = img_name_train_total_shuffle[batch_num * batch_size: (batch_num + 1) * batch_size]
        label_batch[batch_num] = label_train_total_shuffle[batch_num * batch_size: (batch_num + 1) * batch_size]

return img_name_batch, label_batch, img_class, img_name_test_total, label_test_total

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, img_size, img_size, 1])
    
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Add Convolution layer _zy
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    print(conv3.shape)
    
    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    # Apply Dropout
    fc3 = tf.nn.dropout(fc3, dropout)
    
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out


def batch():
    # read jpg
    batch_num = np.random.randint(0, len(img_name_batch))
    img_name = img_name_batch[batch_num]
    label = label_batch[batch_num]
    img_data = np.zeros((len(img_name)*9, img_size * img_size))
    label2 = np.zeros((len(img_name)*9, len(img_class)))
    for i in range(len(img_name)):
        img_full_name = 'scene_categories/' + img_class[label[i]] + '/' + img_name[i]
        img = cv2.imread(img_full_name)[:, :, 0]
        img = cv2.resize(img, (img_size+2, img_size+2))
        d = -1
        for j in range(3):
            for k in range(3):
                d += 1
                img_data_ = np.reshape(img[j:j+img_size, k:k+img_size],(-1,))
                img_data_min = min(img_data_)
                img_data_max = max(img_data_)
                img_data_nor = (img_data_ - img_data_min) / (img_data_max - img_data_min)
                img_data[i*9+d] = img_data_nor
                label2[i*9+d, label[i]] = 1
    #                 img_ = scale(img[j:j+img_size, k:k+img_size])
#                 img_data[i*9+d] = np.reshape(img_,(-1,))
#                 label2[i*9+d, label[i]] = 1
return img_data, label2


def batch_test():
    # read jpg
    img_name = img_name_test
    label = label_test
    img_data = np.zeros((len(img_name), img_size * img_size))
    label2 = np.zeros((len(img_name), len(img_class)))
    for i in range(len(img_name)):
        img_full_name = 'scene_categories/' + img_class[label[i]] + '/' + img_name[i]
        img = cv2.imread(img_full_name)[:, :, 0]
        img = cv2.resize(img, (img_size, img_size))
        img_data_ = np.reshape(img,(-1,))
        img_data_min = min(img_data_)
        img_data_max = max(img_data_)
        img_data_nor = (img_data_ - img_data_min) / (img_data_max - img_data_min)
        img_data[i] = img_data_nor
        label2[i, label[i]] = 1
    #         img = scale(img)
    #         img_data[i] = np.reshape(img, (-1,))
    #         label2[i, label[i]] = 1
    return img_data, label2


img_name_batch, label_batch, img_class, img_name_test, label_test = pre_processing()
img_size = 64

# Training Parameters
learning_rate = 0.001
num_steps = 1500
display_step = 100

# Network Parameters
num_input = img_size * img_size  # MNIST data input (img shape: 28*28)
num_classes = 15  # MNIST total classes (0-9 digits)
dropout = 0.5  # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)




# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])*0.1),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])*0.1),
    # 5X5 conv, 64 inputs, 128 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])*0.1),
    # fully connected, 8*8*128 inputs, 1024 outputs  ## decided by maxpooling k
    'wd1': tf.Variable(tf.random_normal(
                                        [int(img_size/8) * int(img_size/8) * 256, 512])*0.1),
                                        # fully connected, 1024 inputs, 512 outputs
                                        'wd2': tf.Variable(tf.random_normal([512, 256])*0.1),
                                        'wd3': tf.Variable(tf.random_normal([256, 256])*0.1),
                                        # 512 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, num_classes])*0.1)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'bd2': tf.Variable(tf.random_normal([256])),
    'bd3': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                                 logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    print('running cnn')
    sess.run(init)
    
    for step in range(1, num_steps + 1):
        # print('iter: ', step)
        batch_x, batch_y = batch()
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                 Y: batch_y,
                                 keep_prob: 1})
                                 print("Step " + str(step) + ", Minibatch Loss= " + \
                                       "{:.4f}".format(loss) + ", Training Accuracy= " + \
                                       "{:.3f}".format(acc) +'<br>')

print("Optimization Finished!")

test_x, test_y = batch_test()
print('Testing Accuracy:', \
      sess.run(accuracy, feed_dict={X: test_x,
               Y: test_y,
               keep_prob: 1.0}))
          print('<br>learning rate = %.4f, img size = %i, dropout = %.3f' %(learning_rate, img_size, dropout))
