import numpy as np
from mnist import MNIST
import cv2 as cv
import matplotlib.pyplot as plt

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
mnData = MNIST("/home/satyaprakash/Downloads/mnist")
imgs,label = mnData.load_testing()

img = np.array(imgs[0]).reshape(1,28,28)

img_norm = img - img.mean()
img_norm /= img.std()

plt.imshow(img_norm.reshape(28,28))
plt.show()

# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
