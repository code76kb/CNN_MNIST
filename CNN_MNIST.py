from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import os
import time
from Util import *

# Hyper Perameter
no_kernels_0 = 8
no_kernels_1 = 16
kernel_shape = (1,3,3)
kernel_0 = 0
kernel_1 = 0
kernel_0_shape = (no_kernels_0,1,3,3)
kernel_1_shape = (no_kernels_1,no_kernels_0,3,3)
pool_shape = (2,2)
no_conv_layer = 2

no_hidden_layer = 2

no_fc_nodes = (1*7*7)*no_kernels_1
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
def train(test):

    global kernel_0, kernel_1, weight_matrix_1, weight_matrix_2, weight_matrix_3, bais_1, bais_2, bais_3

    epslone = 0.01
    start = int(raw_input(' From ....'))
    till = int(raw_input('Till.......'))
    batchSize = int(raw_input(' batchSize ...'))

    LR = 0.01


    # initialize Perameter
    if(os.path.isfile('/home/satyaprakash/CNN_MNIST/kernel_1.dat')):
        print 'Starting from last, loading old kernels....'

        kernel_0 = np.load('kernel_0.dat')
        kernel_1 = np.load('kernel_1.dat')

        weight_matrix_1 = np.load('weight_matrix_1.dat')
        weight_matrix_2 = np.load('weight_matrix_2.dat')
        weight_matrix_3 = np.load('weight_matrix_3.dat')
        bais_1 = np.load('bais_1.dat')
        bais_2 = np.load('bais_2.dat')
        bais_3 = np.load('bais_3.dat')

    else:
        print 'initializing Perameters........'
        kernel_0 = initializeKernel(kernel_0_shape)
        kernel_1 = initializeKernel(kernel_1_shape)
        weight_matrix_1 = initializeWeights(weight_matrix_1_shape)
        weight_matrix_2 = initializeWeights(weight_matrix_2_shape)
        weight_matrix_3 = initializeWeights(weight_matrix_3_shape)
        bais_1 = np.zeros(bais_1_shape)
        bais_2 = np.zeros(bais_2_shape)
        bais_3 = np.zeros(bais_3_shape)

        print 'kernel_0 shape :',kernel_0.shape
        print 'kernel_1 shape :',kernel_1.shape
        print 'weight_matrix_1 shape :',weight_matrix_1.shape
        print 'weight_matrix_2 shape :',weight_matrix_2.shape
        print 'weight_matrix_3 shape :',weight_matrix_3.shape
        print 'bais_1 shape :',bais_1.shape
        print 'bais_2 shape :',bais_2.shape
        print 'bais_3 shape :',bais_3.shape

    #Preparing input Data
    imgs,labels = mnData.load_training()

    accuracy = 0
    avrCost = 0
    iteration = start - till
    print 'from :',start
    startTime = time.time()
    while start > till:

        i = 0
        total_cost = 0

        while i < batchSize:

            img = np.array(imgs[(start - i) ]).astype(np.float32).reshape((1,28,28))

            img -= (np.mean(img))
            img /= (np.std(img))

            label = labels[ start - i ]
            labels_hot_enc = np.zeros((no_output_nodes,1))
            labels_hot_enc[label,0] = 1

            # Feed Forward

            conv0 = convolution(img,kernel_0)
            relu0 = np.array(conv0)
            relu0 [conv0<0] = 0.01 # Leaky relu
            pooled_0 = pool(relu0,pool_shape)

            conv1 = convolution(pooled_0,kernel_1)
            relu1 = np.array(conv1)
            relu1[conv1<0] = 0.01 # Leaky relu
            pooled_1 = pool(relu1,pool_shape)

            fc = pooled_1.reshape(no_fc_nodes,1)  # flatten img

            hidden_layer_1 = weight_matrix_1.dot(fc) + bais_1
            relu3 = np.array(hidden_layer_1)
            relu3[hidden_layer_1<0] = 0.01 # Leaky Relu

            hidden_layer_2 = weight_matrix_2.dot(relu3) + bais_2
            relu4 = np.array(hidden_layer_2)
            relu4[hidden_layer_2<0] = 0.01 # Leaky Relu

            output_layer = weight_matrix_3.dot(relu4) + bais_3
            probs = Softmax(output_layer) # Softmax

            print 'pridiction :',probs.argmax(),' actual :',label, 'confidance :',probs.max()
            if(probs.argmax() == label):
                accuracy += 1


            cost = cross_entropy_error(probs, labels_hot_enc)
            # print 'Cost :',cost
            total_cost += cost

            # Back Propegation
            dout = probs - labels_hot_enc

            delta_w3 = dout.dot(relu4.T);
            delta_b3 = dout;
            delta_3 = weight_matrix_3.T.dot(dout);

            delta_3[hidden_layer_2 < 0] *= 0.01
            delta_w2 = delta_3.dot(relu3.T);
            delta_b2 = np.sum(delta_3, axis=1).reshape(bais_2.shape)
            delta_2  = weight_matrix_2.T.dot(delta_3)

            delta_2[hidden_layer_1 < 0] *= 0.01
            delta_w1 = delta_2.dot(fc.T)
            delta_b1 = np.sum(delta_2,axis=1).reshape(bais_1.shape)
            delta_1  = weight_matrix_1.T.dot(delta_2)

            #Conv
            delta_1 = delta_1.reshape(pooled_1.shape)
            delta_pooled_1 = d_pool(conv1, delta_1, pool_shape, 2)
            delta_pooled_1[conv1<0] *= 0.01
            delta_kernel_1, delta_conv_1 = d_convolution(pooled_0, delta_pooled_1, kernel_1)

            delta_pooled_0 = d_pool(conv0, delta_conv_1, pool_shape, 2)
            delta_pooled_0[conv0<0] *= 0.01
            delta_kernel_0, delta_out = d_convolution(img, delta_pooled_0, kernel_0)


            # Update weights
            weight_matrix_3 -= (delta_w3 * LR);
            bais_3          -= (delta_b3 * LR);

            weight_matrix_2 -= (delta_w2 * LR);
            bais_2          -= (delta_b2 * LR);

            weight_matrix_1 -= (delta_w1 * LR);
            bais_1          -= (delta_b1 * LR);

            # kernel
            kernel_1        -= (delta_kernel_1 * LR)
            kernel_0        -= (delta_kernel_0 * LR)





            i += 1;

        print 'From :',start ,' Total Cost :',total_cost/batchSize
        avrCost += total_cost

        start -= batchSize #1

    print '\nAvg Cost  :',avrCost/(iteration), '\n Accuracy :',accuracy/iteration
    print '\n took :', ((time.time()-startTime)/60),' mins.'
    save = raw_input('save learning ?')
    if(save == 'y'):
        print 'Saving ...'
        kernel_0.dump('kernel_0.dat')
        kernel_1.dump('kernel_1.dat')
        weight_matrix_1.dump('weight_matrix_1.dat')
        weight_matrix_2.dump('weight_matrix_2.dat')
        weight_matrix_3.dump('weight_matrix_3.dat')
        bais_1.dump('bais_1.dat')
        bais_2.dump('bais_2.dat')
        bais_3.dump('bais_3.dat')


#Show

def show(conv1):
    i,_,_ = conv1.shape

    for k in range(i):
        plt.subplot(i,1,k+1)
        plt.imshow(conv1[k,:,:])

    plt.show()

def showK(conv1):
    i,_,s1,s2 = conv1.shape

    for k in range(i):
        plt.subplot(i,1,k+1)
        plt.imshow((conv1[k,:,:,:]).reshape(s1,s2))

    plt.show()


#Test
def test():

    #Preparing input Data
    imgs,labels = mnData.load_training()

    kernel_1 = np.load('kernel_1.dat')
    kernel_2 = np.load('kernel_2.dat')

    weight_matrix_1 = np.load('weight_matrix_1.dat')
    weight_matrix_2 = np.load('weight_matrix_2.dat')
    weight_matrix_3 = np.load('weight_matrix_3.dat')
    bais_1 = np.load('bais_1.dat')
    bais_2 = np.load('bais_2.dat')
    bais_3 = np.load('bais_3.dat')

    till = int(raw_input('till....'))
    accuracy = 0
    i = 0
    multipal = int(raw_input('from....'))
    while(i < till):
        img = np.array(imgs[multipal+i]).astype(np.float32).reshape((1,28,28))
        label = labels[multipal+i]

        img -= (np.mean(img))
        img /= (np.std(img))

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
        i += 1
        if(probs.argmax() == label):
            accuracy += 1

        # show(conv1)
        # show(conv2)

    print 'test accuracy :',accuracy/till





def start():
    mode = raw_input("press 't' to Train and 'r' to test... ")
    if(mode =='t'):
        # Train
        train(False)
    elif(mode == 'r'):
        # test or run
        train(True)

start()
