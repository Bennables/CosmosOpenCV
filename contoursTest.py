"""
To-Do
add detection for an orange thing in the lane
change direction
yesh
create object to let it stop/continue
make functions for stopping for fixed time, and continuing
clean up code
create boolean for if there's a stop sign
comment for clarity
"""

import cv2 as cv2
from cv2 import imshow
from cv2 import findContours
from cv2 import drawContours
from matplotlib.pyplot import gray
import numpy as np
import lane_functions

def imShow(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def toCanny(img):
    gray_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus_x = cv2.GaussianBlur(gray_x, (5,5), 0)
    canny_x = cv2.Canny(gaus_x, 100, 200)
    return canny_x

def contours(img, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(img, contours, -1, (0,255,0), 3)
    max_len_index = 0
    max_len = 0
    for counter, cont in enumerate(contours):
        if len(cont) > max_len:
            max_len_index = counter
    # print("contours", contours)
    # print('max len index' , max_len_index)
    try:
        x, y, w, h = cv2.boundingRect(contours[max_len_index])

        #relative to shape(shape[1]* .2) <-- sumthin
        if w > int(img.shape[1] * .1):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        return img, x, y, w, h
    except:
        print('img not found')


car = lane_functions(serial_port='COM3')
orig_image = cv2.imread('stop2.jpg')
orig_image = orig_image.copy()
cannied = toCanny(orig_image)
#imShow('canny', cannied)

low = np.uint8([0, 0, 100])
high = np.uint8([30, 30, 255])
mask = cv2.inRange(orig_image, low, high)
imShow('mask', mask)
try:
    stopSign, x, y, w, h = contours(orig_image, mask)
    cv2.putText(stopSign, 'Stop sign detected!', (((x + w) // 2) - 10, y - 20), cv2.FONT_HERSHEY_PLAIN, int(orig_image.shape[1]*.002), (0, 255, 0), int(orig_image.shape[1] * 0.0005))
except:
    pass

imShow('contoured', stopSign) 