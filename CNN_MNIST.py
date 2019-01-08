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
mnData = MNIST("/home/satyaprakash/Downloads/mnist")



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

        print 'kernel_1 shape :',kernel_1.shape
        print 'kernel_2 shape :',kernel_2.shape
        print 'weight_matrix_1 shape :',weight_matrix_1.shape
        print 'weight_matrix_2 shape :',weight_matrix_2.shape
        print 'weight_matrix_3 shape :',weight_matrix_3.shape
        print 'bais_1 shape :',bais_1.shape
        print 'bais_2 shape :',bais_2.shape
        print 'bais_3 shape :',bais_3.shape
    #Preparing input Data
    imgs,labels = mnData.load_training()

    while epoch > 0:

        D_kernel1 =0
        D_kernel2 =0
        D_weight_matrix1 =0
        D_weight_matrix2 =0
        D_weight_matrix3 =0
        D_bais1 =0
        D_bais2 =0
        D_bais3 =0


        while itration > 0:

            img = np.array(imgs[itration]).reshape((1,28,28))
            label = labels[itration]
            labels_hot_enc = np.zeros((no_output_nodes,1))
            labels_hot_enc[label,0] = 1

            # Feed Forward

            conv1 = convolution(img,kernel_1)
            conv1[conv1<0] = 0 # Leaky relu
            pooled_1 = pool(conv1,pool_shape)
            # print 'pooled_1 max :',pooled_1.max()

            conv2 = convolution(pooled_1,kernel_2)
            conv2[conv2<0] = 0 #Leaky relu
            pooled_2 = pool(conv2,pool_shape)
            # print 'pooled_2 max :',pooled_2.max()

            fc = pooled_2.flatten().reshape(no_fc_nodes,1)  # flatten img

            hidden_layer_1 = np.dot(weight_matrix_1, fc) + bais_1
            hidden_layer_1[hidden_layer_1<0]=0 # Leaky Relu
            # print 'layer_1 max:',hidden_layer_1.max()

            hidden_layer_2 = np.dot(weight_matrix_2, hidden_layer_1) + bais_2
            hidden_layer_2[hidden_layer_2<0]=0 # Leaky Relu
            # print 'layer_2 max:',hidden_layer_2.max()

            output_layer = np.dot(weight_matrix_3, hidden_layer_2) + bais_3
            probs = Softmax(output_layer) # Softmax
            print 'pridiction :',probs.argmax(),' actual :',label, 'confidance :',probs.max()

            cost = cross_entropy_error(probs, labels_hot_enc)
            print 'Cost :',cost

            # Back Propegation
            dout = probs - labels_hot_enc

            dedw3 = np.dot(dout, hidden_layer_2.T)
            dedb3 = np.sum(dout,axis=1).reshape(bais_3.shape)
            # print 'dedw3 max :',dedw3.max()

            delta2 = np.dot(weight_matrix_3.T,dout)
            dedw2 = np.dot(delta2, hidden_layer_1.T)
            dedb2 = np.sum(delta2, axis=1).reshape(bais_2.shape)
            # print 'dedw2 max :',dedw2.max()


            delta1 = np.dot(weight_matrix_2.T, delta2)
            dedw1 = np.dot(delta1,fc.T)
            dedb1 = np.sum(delta1,axis=1).reshape(bais_1.shape)
            # print 'dedw1 max :',dedw1.max()

            delta0 = np.dot(weight_matrix_1.T, delta1).reshape(pooled_2.shape)
            # print 'delta 0 :',delta0.shape

            # conv layer-2
            dconv2 = d_pool(conv2,delta0,pool_shape)
            # print 'dconv2 shape :',dconv2.shape
            dconv2[conv2<0] = 0
            # Decone-2
            delta_kernel2, delta_conv2 = d_convolution(pooled_1, dconv2, kernel_2)
            # print 'delta_kernel shape:',delta_kernel2.shape
            # print 'delta_conv2 shape:',delta_conv2.shape
            # print 'pooled 1 shape :',pooled_1.shape

            # conv layer -1
            dconv1 = d_pool(conv1,delta_conv2,pool_shape)
            # print 'dconv1 shape :',dconv1.shape
            dconv1[conv1<0] = 0
            delta_kernel1, delta_conv1 = d_convolution(img, dconv1, kernel_1)
            # print 'delta_kernel1 shape:',delta_kernel1.shape
            # print 'delta_conv1 shape:',delta_conv1.shape
            # print 'img shape :',img.shape

            # Update Perameters
            # kernel_1 = kernel_1 - delta_kernel1
            # kernel_2 = kernel_2 - delta_kernel2
            # weight_matrix_1 = weight_matrix_1 - dedw1
            # weight_matrix_2 = weight_matrix_2 - dedw2
            # weight_matrix_3 = weight_matrix_3 - dedw3
            # bais_1 = bais_1 - dedb1
            # bais_2 = bais_2 - dedb2
            # bais_3 = bais_3 - dedb3

            D_kernel1 += delta_kernel1
            D_kernel2 += delta_kernel2
            D_weight_matrix1 += dedw1
            D_weight_matrix2 += dedw2
            D_weight_matrix3 += dedw3
            D_bais1 += dedb1
            D_bais2 += dedb2
            D_bais3 += dedb3


            # print 'Dkernel_1 shape :',delta_kernel1.shape
            # print 'Dkernel_2 shape :',delta_kernel2.shape
            # print 'Dweight_matrix_1 shape :',dedw1.shape
            # print 'Dweight_matrix_2 shape :',dedw2.shape
            # print 'Dweight_matrix_3 shape :',dedw3.shape
            # print 'Dbais_1 shape :',dedb1.shape
            # print 'Dbais_2 shape :',dedb2.shape
            # print 'Dbais_3 shape :',dedb3.shape

            print 'dkernel_1 max:',delta_kernel1.max(),' min:',delta_kernel1.min()
            print 'dkernel_2 max:',delta_kernel2.max(),' min:',delta_kernel2.min()
            print 'dweight_matrix_1 max:',weight_matrix_1.max(),' min:',weight_matrix_1.min()
            print 'dweight_matrix_2 max:',weight_matrix_2.max(),' min:',weight_matrix_2.min()
            print 'dweight_matrix_3 max:',weight_matrix_3.max(),' min:',weight_matrix_3.min()
            print 'dbais_1 max:',bais_1.max(),' min:',bais_1.min()
            print 'dbais_2 max:',bais_2.max(),' min:',bais_2.min()
            print 'dbais_3 max:',bais_3.max(),' min:',bais_3.min()

            itration -= 1

        # Update Perameters
        kernel_1 -= D_kernel1 / itration
        kernel_2 -= D_kernel2 / itration
        weight_matrix_1 -= D_weight_matrix1 / itration
        weight_matrix_2 -= D_weight_matrix2 / itration
        weight_matrix_3 -= D_weight_matrix3 / itration
        bais_1 -=  D_bais1 / itration
        bais_2 -=  D_bais2 / itration
        bais_3 -=  D_bais3 / itration

        epoch -= 1

def start():
    mode = raw_input("press 't' to Train and 'r' to test... ")

    if(mode =='t'):
        # Train
        train()
    elif(mode == 'r'):
        # test or run
        test()

start()
