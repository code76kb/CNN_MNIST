import numpy as np
from mnist import MNIST
import cv2 as cv
import matplotlib.pyplot as plt


drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

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


img = np.zeros((224,224,1), np.uint8)
cv.namedWindow('Input')
cv.setMouseCallback('Input',interactive_drawing)

while(1):
    cv.imshow('Input',img)
    k=cv.waitKey(1)&0xFF
    if k==27:
        break
cv.destroyAllWindows()


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
