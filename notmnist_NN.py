# This code is implemted in the Tensorflow with one hiddine layer to classifiy the notMNIST dataset
# Code structure is classed based
# Tested and implemented on GTX980 Nvidia Graphic card
# Cuda 9.1 was used to test this code
##############################################################################################################
##############################################################################################################

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os

class notmnist_NN:

    def __init__(self):
        # variable creation and inialization 
        self.im_size = 28  
        self.pixel_depth = 255.0  
        self.valid_data =[]
        self.valid_labels=[]
        self.train_data=[]
        self.train_labels=[]
        self.test_data=[]
        self.test_labels=[]
        self.batch_size = 128
        self.num_steps = 3001
        self.num_labels = 10
        self.h_nodes = 1024
        
    def check_data(self):

        data_file = 'notMNIST.pickle'
        if os.path.exists(data_file):
            print('%s already present - Loading Data from file' % data_file)
            notMNIST = open('notMNIST.pickle', 'rb')
            data = pickle.load(notMNIST)
            notMNIST.close()
            self.train_data = data['train_data']
            self.train_labels = data['train_labels']
            self.valid_data = data['valid_data']
            self.valid_labels = data['valid_labels']
            self.test_data = data['test_data']
            self.test_labels = data['test_labels']
            del data
        else:
            print('Unable to load dataset, data file not exists')
    # formated data to feed into neural network tensor graph
    def reformat(self,dataset, labels):
        dataset = dataset.reshape((-1, self.im_size * self.im_size)).astype(np.float32)
        labels = (np.arange(self.num_labels) == labels[:, None]).astype(np.float32)
        return dataset, labels
  
        #Calcualte the accuracy of the classification 
    def accuracy(self,predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])


    def trainingANDtesting(self):

        self.train_data, self.train_labels = self.reformat(self.train_data, self.train_labels)
        self.valid_data, self.valid_labels = self.reformat(self.valid_data, self.valid_labels)
        self.test_data, self.test_labels = self.reformat(self.test_data, self.test_labels)

        graph = tf.Graph()

        with graph.as_default():

            # creating placeholder for input data to the NN
            tf_train_dataset = tf.placeholder(
                tf.float32, shape=(self.batch_size, self.im_size * self.im_size))
            tf_train_labels = tf.placeholder(
                tf.float32, shape=(self.batch_size, self.num_labels))
            tf_valid_dataset = tf.constant(self.valid_data)
            tf_test_dataset = tf.constant(self.test_data)

            weights_h = tf.Variable(
                tf.truncated_normal([self.im_size * self.im_size, self.h_nodes]))
            biases_h = tf.Variable(tf.zeros([self.h_nodes]))
            # output of hidden layer
            weights_out = tf.Variable(
                tf.truncated_normal([self.h_nodes, self.num_labels]))
            biases_out = tf.Variable(tf.zeros([self.num_labels]))

            # Training 
            logits_h = tf.matmul(tf_train_dataset, weights_h) + biases_h
            logits = tf.matmul(tf.nn.relu(logits_h), weights_out) + biases_out
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.relu(logits)
            valid_prediction = tf.nn.relu(
                tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_h) + biases_h), weights_out) + biases_out)
            test_prediction = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(
                tf_test_dataset, weights_h) + biases_h), weights_out) + biases_out)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("Initialized")
            for step in range(self.num_steps):
        
                offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)

                batch_data = self.train_data[offset:(offset + self.batch_size), :]
                batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
    
                feed_dict = {tf_train_dataset: batch_data,
                            tf_train_labels: batch_labels}
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if step % 500 == 0:
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" %
                        self.accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % self.accuracy(
                        valid_prediction.eval(), self.valid_labels))
            print("Test accuracy with ReLU: %.1f%%" %
                self.accuracy(test_prediction.eval(), self.test_labels))

not_nn = notmnist_NN()
not_nn.check_data()
not_nn.trainingANDtesting()






