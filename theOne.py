import cv2 as cv2
from cv2 import findContours
import numpy as np
from edgeDetection import *
from matplotlib import pyplot as plt

# #makes a heat map for distance
# imgL = cv2.imread('tsukuba_l.png',0)
# imgR = cv2.imread('tsukuba_r.png',0)
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity,'gray')
# plt.show()

  
# find_contours takes an image, and a mask. 
# it finds the contours on the mask and slaps them on the image
def find_contours(img, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    print('hfdkld')
    print(contours[2] )
    x, y, w, h = cv2.boundingRect(cnt)
    print(x, y, w, h)
    cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 2 )
    return mask 



# doing contours(borders, not colors changes(canny))
img = cv2.imread('stop.jpg')
img_copy = np.copy(img)
lower = np.array([0, 0, 255])
upper = np.array([100, 255, 255])
mask = cv2.inRange(img, lower, upper)
show_image('cum', find_contours(img_copy, mask))

print ('hi')



# def detect_colors_in_bounds(img):
#     lower = np.array([0, 0, 255])
#     upper = np.array([100, 255, 255])
#     mask = cv2.inRange(img, lower, upper)
#     show_image('mask', mask)
#     x = find_contours(img, mask)
#     show_image('s', x)
#     points = [[0, 0], [0,img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1],0]]
#     cv2.fillPoly(img, np.int32([points]), color=(0, 255, 0))
    
#     return img
