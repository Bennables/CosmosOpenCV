import cv2 as cv2
import numpy as np
import matplotlib as mpl
from edgeDetection import *


def find_contours(img, mask):
    contours, hierarchy = cv2.findContours(mask,  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img

img = cv2.imread('coloredObjects.jpg')
lower = np.array([0,0,0])
upper = np.array([60, 60, 255])
mask = cv2.inRange(img, lower, upper)
img = find_contours(img, mask)


show_image('idiot', img)
print ('hi')


