#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon., Oct. 14, 2019

@author: young-hwanlee

-> test the autoencoder using MNIST
"""

#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# mnist = tf.keras.datasets.mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

## Set up the seed for reproducibility.
seed = 42
tf.reset_default_graph()
tf.set_random_seed(seed)
np.random.seed(seed)

# cross_entropy_check = "1"   # use the cross entropy
cross_entropy_check = "0"   # don't use the cross entropy

scale_weights = 1e-4
learning_rate = 1e-1
n_epochs = 50
batch_size = 100
# batch_size = 126
epsilon = 1e-35

#%%
## [construction phase]
n_inputs = 28*28
n_hidden1 = 256
n_hidden2 = 256

if cross_entropy_check == "1":
    n_codes = 10
else:
    n_codes = 100

n_outputs = n_inputs

n_train = 20000
X_train = mnist.train.images[:n_train]
X_val = mnist.validation.images
X_test = mnist.test.images

# def OneHotEncoding(y, n_codes):
#     import numpy as np
#     y_temp = np.zeros((y.shape[0], n_codes))
#
#     for j in range(n_codes):
#         for i in range(y.shape[0]):
#             if y[i] == j:
#                 y_temp[i, j] = np.asarray(1, dtype='float64')
#     return y_temp
#
# y_train = OneHotEncoding(mnist.train.labels[:n_train], n_codes)
# y_val = OneHotEncoding(mnist.validation.labels, n_codes)
# y_test = OneHotEncoding(mnist.test.labels, n_codes)

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
# y = tf.placeholder(tf.float32, shape=[None], name="y")
y = tf.placeholder(tf.float32, shape=[None, n_codes], name="y")

b1 = tf.Variable(tf.zeros(shape=[n_hidden1]), dtype=tf.float32)
b2 = tf.Variable(tf.zeros(shape=[n_codes]), dtype=tf.float32)
b3 = tf.Variable(tf.zeros(shape=[n_hidden2]), dtype=tf.float32)
b4 = tf.Variable(tf.zeros(shape=[n_inputs]), dtype=tf.float32)

## He initialization (with any variant of ReLU)
initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
W1 = tf.Variable(initializer(shape=[n_inputs, n_hidden1]), dtype=tf.float32, name='W1')
W2 = tf.Variable(initializer(shape=[n_hidden1, n_codes]), dtype=tf.float32, name='W2')
W3 = tf.Variable(initializer(shape=[n_codes, n_hidden2]), dtype=tf.float32, name='W3')
W4 = tf.Variable(initializer(shape=[n_hidden2, n_outputs]), dtype=tf.float32, name='W4')
# W3 = tf.transpose(W2, name='W3')
# W4 = tf.transpose(W1, name='W4')

with tf.name_scope("encoder"):  # to group related nodes
    hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    codes = tf.matmul(hidden1, W2) + b2

    if cross_entropy_check == "1":
        ## avoid "nan" from "cross_entropy_loss" (caused by very small number from "softmax_function")
        softmax_function = tf.nn.softmax(codes) + epsilon
    else:
        codes_with_activation_fnc = tf.nn.relu(codes)

with tf.name_scope("decoder"):  # to group related nodes
    if cross_entropy_check == "1":
        # hidden2 = tf.matmul(softmax_function, W3) + b3
        hidden2 = tf.nn.relu(tf.matmul(softmax_function, W3) + b3)
    else:
        hidden2 = tf.nn.relu(tf.matmul(codes_with_activation_fnc, W3) + b3)

    X_prime = tf.matmul(hidden2, W4) + b4

with tf.name_scope("loss"):  # to group related nodes
    if cross_entropy_check == "1":
        ## Minimize error using cross entropy
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(softmax_function), reduction_indices=1),
                                            name="cross_entropy_loss")   # reduction_indices: The old (deprecated) name for axis

    reg_loss = tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) \
               + tf.reduce_sum(tf.square(W3)) + tf.reduce_sum(tf.square(W4))

    # loss = tf.reduce_mean(tf.square(X - X_prime), name="loss")
    loss = tf.add(tf.reduce_mean(tf.square(X - X_prime)), scale_weights * reg_loss, name="loss")

with tf.name_scope("train"):  # to group related nodes
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):  # to group related nodes
    if cross_entropy_check == "1":
        ## cast: to cast a tensor to a new type.
        # correct = tf.nn.in_top_k(softmax_function, y, 1)
        correct = tf.equal(tf.argmax(softmax_function, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

#%%
init = tf.global_variables_initializer()
saver = tf.train.Saver()  # to save the trained model parameters to disk

if cross_entropy_check == "1":
    acc_train_hist, acc_val_hist = [], []
loss_train_hist, loss_val_hist = [], []

with tf.Session() as sess:
    init.run()

    for epoch in trange(n_epochs):
        # for iteration in range(int(mnist.train.num_examples // batch_size)):
        #     X_batch, y_batch = mnist.train.next_batch(batch_size)
        for iteration in range(int(np.shape(X_train)[0] // batch_size)):
            X_batch = X_train[iteration * batch_size:(iteration + 1) * batch_size]
            # y_batch = OneHotEncoding(y_train[iteration * batch_size:(iteration + 1) * batch_size], n_codes)

            # sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(training_op, feed_dict={X: X_batch})

            # if cross_entropy_check == "1":
            #     acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            #     acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
            #     acc_train_hist.append(acc_train)
            #     acc_val_hist.append(acc_val)

            # loss_train = loss.eval(feed_dict={X: X_batch, y: y_batch})
            # loss_val = loss.eval(feed_dict={X: X_val, y: y_val})
            loss_train = loss.eval(feed_dict={X: X_batch})
            loss_val = loss.eval(feed_dict={X: X_val})
            loss_train_hist.append(loss_train)
            loss_val_hist.append(loss_val)

            if iteration % 100 == 0:
                print("\nepoch :", epoch, "\titeration :", iteration)
                # if cross_entropy_check == "1":
                #     print("acc :", acc_val, "\nloss :", loss_val)
                # else:
                #     print("loss :", loss_val)
                print("loss :", loss_val)

    # X_prime_check = X_prime.eval(feed_dict={X: X_val, y: y_val})
    X_prime_check = X_prime.eval(feed_dict={X: X_val})

#%%
print("MSE : ", np.sum(np.square(X_val - X_prime_check))/(np.shape(X_val)[0]*np.shape(X_val)[1]))

if cross_entropy_check == "1":
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(loss_train_hist)
    plt.plot(loss_val_hist)
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # plt.figure(2)
    plt.subplot(2,1,2)
    plt.plot(acc_train_hist)
    plt.plot(acc_val_hist)
    # plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    # plt.legend(['train', 'validation'], loc='lower right')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
else:
    plt.figure()
    plt.plot(loss_train_hist)
    plt.plot(loss_val_hist)
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

#%%
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_val[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_prime_check[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#%%
# np.savetxt('X_prime_check_10', X_prime_check)
# np.savetxt('X_prime_check_30', X_prime_check)
# np.savetxt('X_prime_check_50', X_prime_check)
# np.savetxt('X_prime_check_100', X_prime_check)

X_prime_check_10 = np.loadtxt('X_prime_check_10')
X_prime_check_30 = np.loadtxt('X_prime_check_30')
X_prime_check_50 = np.loadtxt('X_prime_check_50')

n = 10  # how many digits we will display
plt.figure(figsize=(50, 4))
for i in range(n):
    # display original
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(X_val[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display codes10 reconstruction
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(X_prime_check_10[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display codes30 reconstruction
    ax = plt.subplot(5, n, i + 1 + 2*n)
    plt.imshow(X_prime_check_30[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display codes50 reconstruction
    ax = plt.subplot(5, n, i + 1 + 3*n)
    plt.imshow(X_prime_check_50[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display codes100 reconstruction
    ax = plt.subplot(5, n, i + 1 + 4*n)
    plt.imshow(X_prime_check[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


