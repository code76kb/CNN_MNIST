import numpy as np
from mnist import MNIST
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib


drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

img = cv.imread('/home/satyaprakash/Pictures/d.jpg')
img = cv.resize(img,(300,300))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = gray.reshape((300,300,1))
#gray = gray.reshape((2500,1))

# Data for plotting
# p = np.arange(0,2500)
#
# print 'img shape :',gray.shape,' p :' ,p.shape
# fig, ax = plt.subplots()
# ax.plot(p, gray)
# ax.set(xlabel='gray', ylabel='pix',
#        title='folks')
# ax.grid()
# plt.show()


def convX(img):
    print 'img shape :',img.shape
    kernel = np.array( [[0,1,0],
                        [1,1,1],
                        [0,1,0]] )
    print "kernel shape :",kernel.shape
    threshold = 150
    h,w,c= img.shape
    k_size = 3
    padding = (k_size-1)/2
    output = np.zeros(img.shape)
    padded = np.zeros( (h+(padding*2),w+(padding*2),c))
    padded[padding:h+padding, padding:w+padding,:] = img
    for z in range(c):
        for x in range(w):
            for y in range(h):
                window =  (padded[y:y+k_size , x:x+k_size, z]) * kernel
                # print 'shape of window :',window.shape
                if(np.count_nonzero(window) > 5):
                    print "window over count ..";
                sub = window[:,:] - img[y,x,z]
                sub[sub<0]=0
                delta  = np.sqrt(np.square( sub ))
                # if(np.count_nonzero(delta) > 5):
                #     print 'center :',img[y,x,z]
                #     print "delta over count .."
                #     print 'shape of window :',window
                #     return
                delta[delta[:,:] > threshold] = 0
                count = np.count_nonzero(delta[:,:])
                if(count > 0):
                    output[y,x,z] = np.sum(delta[:,:]) / 9
                # else:
                #     output[y,x,z] = 0
    return output

# img2 = convX(gray)
#
# cv.namedWindow('Input')
# while(1):
#     cv.imshow('Input',img2)
#     k=cv.waitKey(1)&0xFF
#     if k==27:
#         break
# cv.destroyAllWindows()

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing, mode

    if event==cv.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv.line(img,(ix,iy),(x,y),(255,255,255),3)
                ix,iy=x,y
    elif event==cv.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv.line(img,(x,y),(x,y),(255,255,255),3)


# img = np.zeros((224,224,1), np.uint8)
# cv.namedWindow('Input')
# cv.setMouseCallback('Input',interactive_drawing)
#
# while(1):
#     cv.imshow('Input',img)
#     k=cv.waitKey(1)&0xFF
#     if k==27:
#         break
# cv.destroyAllWindows()


# a = np.random.uniform(size=(2,3))
# b = np.random.normal(size=(3,4))
#
# print 'a :',a, '\nb:',b
# x = np.array([0,-199,9,6000,2,0.76,0.12])
#
# xmean = x.mean()
# xstd = x.std()
#
# # Normalize
# x -= xmean
# x /= xstd
#
# print '\n xMean:',xmean,' \nxstd:',xstd
#
# print '\n x normalize :',x
# mnData = MNIST("/home/satyaprakash/Downloads/mnist")
# imgs,label = mnData.load_testing()

# img = np.array(imgs[0]).reshape(1,28,28)

# img_norm = img - img.mean()
# img_norm /= img.std()
# img_norm = img / img.max()

# print 'norm min max:',img_norm.min(),' :: ',img_norm.max()
# plt.imshow(img_norm.reshape(28,28))
# plt.show()

# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
