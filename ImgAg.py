from __future__ import division

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
import time

rawImg = cv.imread("/home/rahul/Downloads/images.png")
rows,cols,c = rawImg.shape
print("input shape",rawImg.shape)
def showImg(img):
    cv.namedWindow('Raw')
    while(1):
        cv.imshow('Raw',img)
        k=cv.waitKey(1)&0xFF
        if k==27:
            break
    cv.destroyAllWindow()

def plot(imgs):
    plt.subplot(121),plt.imshow(imgs[0]),plt.title('Input')
    plt.subplot(122),plt.imshow(imgs[1]),plt.title('Output')
    plt.show()

def transForm(rawImg,outputShape):
    # startTime = time.time()
    # tx = random.randint(0,rangX)
    # ty = random.randint(0,rangY)
    # print("translate at :",tx,ty)

    # pts1 = np.float32([[50,50],[200,50],[50,200]])
    # pts2 = np.float32([[10,100],[200,50],[100,250]])
    rangX = outputShape[0]
    rangY = outputShape[1]
    padd  = 2 
    scale = random.randint(int(-rangX/2),int((-rangX/100)*20))
    angle = random.randint(-20,20)

    scaleMatrix =np.float32( [[1,1],[-1,1],
                              [1,-1],[-1,-1]]
                            )

    txMatrix = np.float32( [[1,0],[1,0],
                           [1,0],[1,0]]
                         )

    tyMatrix = np.float32( [[0,1],[0,1],
                           [0,1],[0,1]]
                         )

    tx = random.SystemRandom().randint(0-scale, rangX+scale)
    ty = random.SystemRandom().randint(0-scale, rangY+scale)

    #print("translate at :",tx,ty)
    #print("scale :",scale)

    pts3 = np.float32([[ padd, padd ],[ outputShape[0]-padd, padd ], [ padd, outputShape[1]-padd ], [ outputShape[0]-padd, outputShape[1]-padd ]])
    scaleBy = scale * scaleMatrix
    # pts4 = pts3 - scaleBy

    translate = (txMatrix * tx) + (tyMatrix * ty)
    pts4 = translate + scaleBy

    center=tuple(np.array(outputShape)/2)

    rot_mat = cv.getRotationMatrix2D(center,angle,1.0)
    M2 = cv.getPerspectiveTransform(pts3,pts4)

    output_1 = cv.warpPerspective(rawImg, M2, outputShape)
    output = cv.warpAffine(output_1, rot_mat, outputShape)

    return output

# dst = transForm(rows,cols, (rows,cols))

# plot([rawImg,dst])
