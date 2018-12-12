import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import time


Train_labels = np.loadtxt("/home/amyassient/PycharmProjects/HCNA-AI/WISDM_ar_v1.1/Train_labels.txt")
Train_data = np.loadtxt("/home/amyassient/PycharmProjects/HCNA-AI/WISDM_ar_v1.1/Train_data.txt")
Train_data = np.reshape(Train_data,(600,1,100,3))
Train_labels = np.reshape(Train_labels,(600,6))

Test_labels = np.loadtxt("/home/amyassient/PycharmProjects/HCNA-AI/WISDM_ar_v1.1/Test_labels.txt")
Test_data = np.loadtxt("/home/amyassient/PycharmProjects/HCNA-AI/WISDM_ar_v1.1/Test_data.txt")
Test_data = np.reshape(Test_data,(60,1,100,3))
Test_labels = np.reshape(Test_labels,(60,6))

input_height = 1
input_width = 100
num_labels = 6
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 500

def CNN(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[1,10,3,60],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[60],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,1,20,1],strides=[1,1,2,1],padding="SAME")
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight",[1,6,60,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias",[10],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1],padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.variable_scope('layer4-fc1'):
        pool_shape = relu2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(relu2, [-1, nodes])

        fc1_weights = tf.get_variable("weight",[nodes,1000],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1000], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.tanh(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer5-fc2'):
        fc2_weights = tf.get_variable("weight", [1000,6],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [6], initializer=tf.constant_initializer(0.1))
        logits = tf.nn.softmax(tf.matmul(fc1, fc2_weights) + fc2_biases)
    return logits

X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])
regularizer = tf.contrib.layers.l2_regularizer(0.001)
y_ = CNN(X, False, regularizer)
loss = -tf.reduce_sum(Y*tf.log(y_))
#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=y_)
#loss =  tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    tf.initialize_all_variables().run()

    for epoch in range(training_epochs):
        _, tra_ = session.run([optimizer, accuracy], feed_dict={X: Train_data, Y: Train_labels})
        testa = session.run(accuracy, feed_dict={X: Test_data, Y: Test_labels})
        print("Epoch: ", epoch, " Training Accuracy: ",session.run(accuracy, feed_dict={X: Train_data, Y: Train_labels}))
        print("Testing Accuracy:", session.run(accuracy, feed_dict={X: Test_data, Y: Test_labels}))
