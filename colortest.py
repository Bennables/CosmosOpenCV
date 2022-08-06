import cv2


import cv2 as cv2
from cv2 import imshow
from cv2 import COLOR_HLS2BGR
import numpy as np
def imShow(img, name='fd'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

x = cv2.imread('nature.jpg')


x = np.copy(x)
xx = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
xy = cv2.cvtColor(x, cv2.COLOR_HLS2BGR)
y = cv2.cvtColor(x, cv2.COLOR_BGR2HLS)
z = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

imShow(x)
# imShow(xx, 'dfasdf  ')
# imShow(xy , 'hlstobgr')
imShow(y, 'hls')
imShow(z, 'hsv')


