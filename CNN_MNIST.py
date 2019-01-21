from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import os

from Util import *

# Hyper Perameter
no_kernels_1 = 18
no_kernels_2 = 10
kernel_shape = (1,3,3)
kernel_1 = 0
kernel_2 = 0
kernel_1_shape = (no_kernels_1,1,3,3)
kernel_2_shape = (no_kernels_2,no_kernels_1,3,3)
pool_shape = (2,2)
no_conv_layer = 2


no_hidden_layer = 2

no_fc_nodes = (1*7*7)*no_kernels_2
no_hidden_nodes_1 = 200
no_hidden_nodes_2 = 100
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
# mnData = MNIST("/mnt/66C2AAD8C2AAABAD/ML_init/DataSets/mnist")



# Train
def train():

    global kernel_1, kernel_2, weight_matrix_1, weight_matrix_2, weight_matrix_3, bais_1, bais_2, bais_3

    epslone = 0.0001
    epoch = int(raw_input(' Epoch ....'))
    till = int(raw_input('Till.......'))
    batchSize = int(raw_input(' batchSize ...'))

    LR = 0.0001
    beta = 0.9

    vk1 = 0
    vk2 = 0
    vw1 = 0
    vw2 = 0
    vw3 = 0
    vb1 = 0
    vb2 = 0
    vb3 = 0

    # initialize Perameter
    if(os.path.isfile('/home/satyaprakash/CNN_MNIST/kernel_1.dat')):
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

    while epoch > till:

        i = 0
        total_cost = 0
        dk1_ = 0
        dk2_ = 0
        dw1_ = 0
        dw2_ = 0
        dw3_ = 0
        db1_ = 0
        db2_ = 0
        db3_ = 0


        while i < batchSize:

            img = np.array(imgs[i + (epoch * batchSize) ]).astype(np.float32).reshape((1,28,28))

            img -= (np.mean(img))
            img /= (np.std(img))

            label = labels[i + (epoch * batchSize)]
            labels_hot_enc = np.zeros((no_output_nodes,1))
            labels_hot_enc[label,0] = 1

            # Feed Forward

            relu1 = conv1 = convolution(img,kernel_1)
            relu1 [conv1<0] = 0.01 # Leaky relu
            pooled_1 = pool(relu1,pool_shape)
            # print 'pooled_1 max :',pooled_1.max()

            relu2 = conv2 = convolution(pooled_1,kernel_2)
            relu2[conv2<0] = 0.01 #Leaky relu
            pooled_2 = pool(relu2,pool_shape)
            # print 'pooled_2 max :',pooled_2.max()

            fc = pooled_2.reshape(no_fc_nodes,1)  # flatten img

            relu3 = hidden_layer_1 = weight_matrix_1.dot(fc) + bais_1
            relu3[hidden_layer_1<0] = 0.01 # Leaky Relu
            # print 'layer_1 max:',hidden_layer_1.max()

            relu4 = hidden_layer_2 = weight_matrix_2.dot(relu3) + bais_2
            relu4[hidden_layer_2<0] = 0.01 # Leaky Relu
            # print 'layer_2 max:',hidden_layer_2.max()

            output_layer = weight_matrix_3.dot(relu4) + bais_3

            probs = Softmax(output_layer) # Softmax

            print 'pridiction :',probs.argmax(),' actual :',label, 'confidance :',probs.max()

            cost = cross_entropy_error(probs, labels_hot_enc)
            # print 'Cost :',cost
            total_cost += cost

            # Back Propegation
            dout = probs - labels_hot_enc

            # print 'probs :',probs,'\n label :',labels_hot_enc,' \n dout :',dout

            # print '\ndout max :',dout.max(),' min:', dout.min()
            dedw3 = dout.dot(relu4.T)
            dedb3 = np.sum(dout,axis=1).reshape(bais_3.shape)
            # print 'dedw3 max :',dedw3.max(),' min:', dedw3.min(),' i :',i

            delta2 = weight_matrix_3.T.dot(dout)
            delta2[hidden_layer_2<0] *= 0.01
            dedw2 = delta2.dot(relu3.T)
            dedb2 = np.sum(delta2, axis=1).reshape(bais_2.shape)
            # print 'dedw2 max :',dedw2.max(),' min:',dedw2.min()


            delta1 = weight_matrix_2.T.dot(delta2)
            delta1[hidden_layer_1 < 0] *= 0.01
            dedw1 = delta1.dot(fc.T)
            dedb1 = np.sum(delta1,axis=1).reshape(bais_1.shape)
            # print 'dedw1 max :',dedw1.max(),' min:',dedw1.min()

            delta0 = np.dot(weight_matrix_1.T, delta1).reshape(pooled_2.shape)
            # print 'delta 0 :',delta0.shape

            # conv layer-2
            dconv2 = d_pool(relu2,delta0,pool_shape)
            # print 'dconv2 shape :',dconv2.shape
            dconv2[conv2<0] *= 0.01
            # Decone-2
            delta_kernel2, delta_conv2 = d_convolution(pooled_1, dconv2, kernel_2)
            # print 'delta_kernel shape:',delta_kernel2.shape
            # print 'delta_conv2 shape:',delta_conv2.shape
            # print 'pooled 1 shape :',pooled_1.shape

            # conv layer -1
            dconv1 = d_pool(relu1, delta_conv2, pool_shape)
            # print 'dconv1 shape :',dconv1.shape
            dconv1[conv1<0] *= 0.01
            delta_kernel1, delta_conv1 = d_convolution(img, dconv1, kernel_1)
            # print 'delta_kernel1 shape:',delta_kernel1.shape
            # print 'delta_conv1 shape:',delta_conv1.shape
            # print 'img shape :',img.shape

            # fdw2 = dedw2.flatten()
            # fdw3 = dedw3.flatten()
            #

            # dk1_ += delta_kernel1
            # dk2_ += delta_kernel2
            # dw1_ += dedw1
            # dw2_ += dedw2
            # dw3_ += dedw3
            # db1_ += dedb1
            # db2_ += dedb2
            # db3_ += dedb3

            # Update Perameters
            # kernel_1 = kernel_1 - (delta_kernel1)*0.01
            # kernel_2 = kernel_2 - (delta_kernel2)*0.01
            # weight_matrix_1 = weight_matrix_1 - (dedw1)*0.01
            # weight_matrix_2 = weight_matrix_2 - (dedw2)*0.01
            # weight_matrix_3 = weight_matrix_3 - (dedw3)*0.01
            # bais_1 = bais_1 - (dedb1)*0.01
            # bais_2 = bais_2 - (dedb2)*0.01
            # bais_3 = bais_3 - (dedb3)*0.01

            # print '\nkernel_1 max :',kernel_1.max(),' min :',kernel_1.min()
            # print 'kernel_2 max :',kernel_2.max(),' min :',kernel_2.min()
            # print 'weight_matrix_1 max :',weight_matrix_1.max(),' min :',weight_matrix_1.min()
            # print 'weight_matrix_2 max :',weight_matrix_2.max(),' min :',weight_matrix_2.min()
            # print 'weight_matrix_3 max :',weight_matrix_3.max(),' min :',weight_matrix_3.min()
            # print 'bais_1 max :',bais_1.max(),' min :',bais_1.min()
            # print 'bais_2 max :',bais_2.max(),' min :',bais_2.min()
            # print 'bais_3 max :',bais_3.max(),' min :',bais_3.min()


            i += 1
        # Update Perameters
        # vk1 =  (dk1_/batchSize)
        # vk2 =  (dk2_/batchSize)
        # vw1 =  (dw1_/batchSize)
        # vw2 =  (dw2_/batchSize)
        # vw3 =  (dw3_/batchSize)
        # vb1 =  (db1_/batchSize)
        # vb2 =  (db2_/batchSize)
        # vb3 =  (db3_/batchSize)

        kernel_1 -= delta_kernel1 * LR
        kernel_2 -= delta_kernel2 * LR
        weight_matrix_1 -= dedw1 * LR
        weight_matrix_2 -= dedw2 * LR
        weight_matrix_3 -= dedw3 * LR
        bais_1 -= dedb1 * LR
        bais_2 -= dedb2 * LR
        bais_3 -= dedb3 * LR

        print 'Epoch :',epoch ,' Total Cost :',total_cost/batchSize

        epoch -= 1

    kernel_1.dump('kernel_1.dat')
    kernel_2.dump('kernel_2.dat')
    weight_matrix_1.dump('weight_matrix_1.dat')
    weight_matrix_2.dump('weight_matrix_2.dat')
    weight_matrix_3.dump('weight_matrix_3.dat')
    bais_1.dump('bais_1.dat')
    bais_2.dump('bais_2.dat')
    bais_3.dump('bais_3.dat')

def start():
    mode = raw_input("press 't' to Train and 'r' to test... ")
    if(mode =='t'):
        # Train
        train()
    elif(mode == 'r'):
        # test or run
        test()

start()
