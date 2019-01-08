from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import os

from Util import *

# Hyper Perameter
no_kernels_1 = 8
no_kernels_2 = 8
kernel_shape = (1,3,3)
kernel_1 = 0
kernel_2 = 0
kernel_1_shape = (no_kernels_1,1,3,3)
kernel_2_shape = (no_kernels_2,no_kernels_1,3,3)
pool_shape = (2,2)
no_conv_layer = 2


no_hidden_layer = 2

no_fc_nodes = (1*7*7)*no_kernels_2
no_hidden_nodes_1 = 80
no_hidden_nodes_2 = 60
no_output_nodes = 10

weight_matrix_1_shape = (no_hidden_nodes_1 , no_fc_nodes)
weight_matrix_2_shape = (no_hidden_nodes_2 , no_hidden_nodes_1)
weight_matrix_3_shape = (no_output_nodes , no_hidden_nodes_2)

bais_1_shape = (no_hidden_nodes_1, 1)
bais_2_shape = (no_hidden_nodes_2, 1)
bais_3_shape = (no_output_nodes, 1)

weight_matrix_1 = 0
weight_matrix_2 = 0
weight_matrix_3 = 0
bais_1 = 0
bais_2 = 0
bais_3 = 0

# load input
mnData = MNIST("/mnt/66C2AAD8C2AAABAD/ML_init/DataSets/mnist")



# Train
def train():

    global kernel_1, kernel_2, weight_matrix_1, weight_matrix_2, weight_matrix_3, bais_1, bais_2, bais_3

    epoch = int(raw_input(' Epoch ....'))
    itration = int(raw_input(' Itration ...'))

    # initialize Perameter
    if(os.path.isfile('/mnt/66C2AAD8C2AAABAD/ML_init/CNN/*.dat')):
        print 'Starting from last, loading old kernels....'
        kernel_1 = np.load('kernel_1.dat')
        kernel_2 = np.load('kernel_2.dat')

        weight_matrix_1 = np.load('weight_matrix_1.dat')
        weight_matrix_2 = np.load('weight_matrix_2.dat')
        weight_matrix_3 = np.load('weight_matrix_3.dat')
        bais_1 = np.load('bais_1.dat')
        bais_2 = np.load('bais_2.dat')
        bais_3 = np.load('bais_3.dat')

    else:
        print 'initializing Perameters........'
        kernel_1 = initializeKernel(kernel_1_shape)
        kernel_2 = initializeKernel(kernel_2_shape)
        weight_matrix_1 = initializeWeights(weight_matrix_1_shape)
        weight_matrix_2 = initializeWeights(weight_matrix_2_shape)
        weight_matrix_3 = initializeWeights(weight_matrix_3_shape)
        bais_1 = np.zeros(bais_1_shape)
        bais_2 = np.zeros(bais_2_shape)
        bais_3 = np.zeros(bais_3_shape)


    #Preparing input Data
    imgs,labels = mnData.load_training()
    img = np.array(imgs[0]).reshape((1,28,28))
    label = labels[0]
    labels_hot_enc = np.zeros((no_output_nodes,1))
    labels_hot_enc[label,0] = 1

    # Feed Forward

    conv1 = convolution(img,kernel_1)
    conv1[conv1<0] = 0.01 # Leaky relu
    pooled_1 = pool(conv1,pool_shape)
    # print 'pooled_1 max :',pooled_1.max()

    conv2 = convolution(pooled_1,kernel_2)
    conv2[conv2<0] = 0.01 #Leaky relu
    pooled_2 = pool(conv2,pool_shape)
    # print 'pooled_2 max :',pooled_2.max()

    fc = pooled_2.flatten().reshape(no_fc_nodes,1)  # flatten img

    hidden_layer_1 = np.dot(weight_matrix_1, fc) + bais_1
    hidden_layer_1[hidden_layer_1<0]=0.01 # Leaky Relu
    # print 'layer_1 max:',hidden_layer_1.max()

    hidden_layer_2 = np.dot(weight_matrix_2, hidden_layer_1) + bais_2
    hidden_layer_2[hidden_layer_2<0]=0.01 # Leaky Relu
    # print 'layer_2 max:',hidden_layer_2.max()

    output_layer = np.dot(weight_matrix_3, hidden_layer_2) + bais_3
    probs = Softmax(output_layer) # Softmax
    print 'output_layer max at:',probs.argmax(), ' max:',probs.max()

    cost = cross_entropy_error(probs, labels_hot_enc)
    print 'Cost :',cost

    # Back Propegation
    dout = probs - labels_hot_enc

    dedw3 = np.dot(dout, hidden_layer_2.T)
    dedb3 = np.sum(dout,axis=1).reshape(bais_3.shape)
    print 'dedw3 max :',dedw3.max()

    delta2 = np.dot(weight_matrix_3.T,dout)
    dedw2 = np.dot(delta2, hidden_layer_1.T)
    dedb2 = np.sum(delta2, axis=1).reshape(bais_2.shape)
    print 'dedw2 max :',dedw2.max()


    delta1 = np.dot(weight_matrix_2.T, delta2)
    dedw1 = np.dot(delta1,fc.T)
    dedb1 = np.sum(delta1,axis=1).reshape(bais_1.shape)
    print 'dedw1 max :',dedw1.max()

    delta0 = np.dot(weight_matrix_1.T, delta1).reshape(pooled_2.shape)
    print 'delta 0 :',delta0.shape

    dconv2 = d_pool(conv2,delta0,pool_shape)
    print 'dconv2 shape :',dconv2.shape
    dconv2[conv2<0] *= 0.01
    





def start():
    mode = raw_input("press 't' to Train and 'r' to test... ")

    if(mode =='t'):
        # Train
        train()
    elif(mode == 'r'):
        # test or run
        test()

start()
