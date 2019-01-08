import numpy as np


def initializeKernel(kernel_size):
    scale = 1.0
    stddev = scale / np.sqrt(np.prod(kernel_size))
    return np.random.normal(loc =0 , scale = stddev, size = kernel_size)

def initializeWeights(weight_matrix_size):
    return np.random.standard_normal(size=weight_matrix_size) * 0.01

def convolution(img, kernel):
    strid = 1

    (no_kernels, no_channels_k, kernel_size, _) = kernel.shape
    no_channels, img_size, _ = img.shape

    padding = int( (kernel_size-1)/2 )

    output_size = int( ((img_size - kernel_size + (2*padding) ) / strid) + 1)

    padding_shape = (no_channels, img_size+(padding*2) , img_size+(padding*2) )
    padded_img = np.zeros(padding_shape)

    padded_img [ :, padding:img_size+padding ,padding:img_size+padding] = img

    assert no_channels == no_channels_k, "Image channels and kernel channels mismatch"

    output = np.zeros( (no_kernels, output_size, output_size) )



    for current_K in range(no_kernels):
        for x in range(0,img_size):
            for y in range(0,img_size):
                output[current_K, x, y] = np.sum(padded_img[:, x:x+kernel_size, y:y+kernel_size] * kernel[current_K])

    return output

def d_convolution(conv, delta, kernel):

    (no_kernels, no_channels_k, kernel_size, _) = kernel.shape
    no_channels, img_size, _ = conv.shape

    stride = 1
    padding = int((kernel_size-1)/2)
    output_kernel = np.zeros((kernel.shape))


    padded_conv = np.zeros( (no_channels, img_size +(2*padding), img_size+(2*padding)) )
    padded_conv[:,padding : img_size + padding,  padding:img_size+padding] = conv

    delta_output = np.zeros((padded_conv.shape))

    for k in range(no_kernels):
        for  x in range(0,img_size):
            for y in range(0,img_size):
                output_kernel[k] += delta[k,x,y] * padded_conv[:, x:x+kernel_size, y:y+kernel_size]
                delta_output[:, x:x+kernel_size, y:y+kernel_size] += delta[k,x,y] * kernel[k]

    return output_kernel,delta_output
    #delta I
    # for c in range(0,no_channels):
    #     for x in range(0,img_size):
    #         for y in range(0,img_size):
    #             output_delta[c: x, y] = delta[]




def pool(img, kernel):
    stride = 2
    no_kernels,img_size,_ = img.shape
    kernel_size,_ = kernel
    s = ((img_size - kernel_size)/stride) + 1

    output = np.zeros((no_kernels, s, s))

    for k in range(0,no_kernels):
        Ix = x = 0
        while Ix <= img_size-stride:
            Iy = y = 0
            while Iy <= img_size-stride:
                output[k,x,y] = np.max(img[k, Ix:Ix+kernel_size, Iy:Iy+kernel_size])
                Iy += stride
                y  += 1
            Ix += stride
            x  += 1

    return output

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def d_pool(conv, delta, pool_shape, stride=2):
    output_error_map = np.zeros(conv.shape)
    pool_size,_ = pool_shape
    no_channels, img_size, _ = conv.shape
    for k in range(no_channels):
        Ix = x = 0
        while Ix <= img_size - pool_size:
            Iy = y = 0
            while Iy <= img_size - pool_size:
                (a,b) = nanargmax(conv[k, Ix: Ix+pool_size, Iy:Iy+pool_size ])
                output_error_map[k, Ix+a, Iy+b] = delta[k,x,y]

                Iy += stride
                y += 1

            Ix += stride
            x += 1
    return output_error_map

def Softmax(X):
    exp = np.exp(X)
    return exp / exp.sum()

def cross_entropy_error(probs,labels):
    return -np.sum(labels * np.log(probs))
