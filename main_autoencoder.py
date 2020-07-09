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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

## Set up the seed for reproducibility.
seed = 42
tf.reset_default_graph()
tf.set_random_seed(seed)
np.random.seed(seed)

scale_weights = 1e-4
learning_rate = 1e-1
n_epochs = 50
batch_size = 100

#%%
## [construction phase]
n_inputs = 28*28
n_hidden1 = 256
n_hidden2 = 256

n_codes = 100
n_outputs = n_inputs

n_train = 20000
X_train = mnist.train.images[:n_train]
X_val = mnist.validation.images
X_test = mnist.test.images

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
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

with tf.name_scope("encoder"):  # to group related nodes
    hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    codes = tf.matmul(hidden1, W2) + b2
    codes_with_activation_fnc = tf.nn.relu(codes)

with tf.name_scope("decoder"):  # to group related nodes
    hidden2 = tf.nn.relu(tf.matmul(codes_with_activation_fnc, W3) + b3)
    X_prime = tf.matmul(hidden2, W4) + b4

with tf.name_scope("loss"):  # to group related nodes
    reg_loss = tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) \
               + tf.reduce_sum(tf.square(W3)) + tf.reduce_sum(tf.square(W4))
    loss = tf.add(tf.reduce_mean(tf.square(X - X_prime)), scale_weights * reg_loss, name="loss")

with tf.name_scope("train"):  # to group related nodes
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

#%%
init = tf.global_variables_initializer()
saver = tf.train.Saver()  # to save the trained model parameters to disk

if cross_entropy_check == "1":
    acc_train_hist, acc_val_hist = [], []
loss_train_hist, loss_val_hist = [], []

with tf.Session() as sess:
    init.run()

    for epoch in trange(n_epochs):
        for iteration in range(int(np.shape(X_train)[0] // batch_size)):
            X_batch = X_train[iteration * batch_size:(iteration + 1) * batch_size]
            sess.run(training_op, feed_dict={X: X_batch})

            loss_train = loss.eval(feed_dict={X: X_batch})
            loss_val = loss.eval(feed_dict={X: X_val})
            loss_train_hist.append(loss_train)
            loss_val_hist.append(loss_val)

            if iteration % 100 == 0:
                print("\nepoch :", epoch, "\titeration :", iteration)
                print("loss :", loss_val)

    X_prime_check = X_prime.eval(feed_dict={X: X_val})

#%%
print("MSE : ", np.sum(np.square(X_val - X_prime_check))/(np.shape(X_val)[0]*np.shape(X_val)[1]))

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

fig.text(0.475, 0.85, 'Original Figures', va='center')
fig.text(0.45, 0.425, 'Reconstructed Figures', va='center')
    
plt.show()


