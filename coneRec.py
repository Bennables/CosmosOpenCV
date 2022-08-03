"""
change direction
comment it up
"""

import cv2 as cv2
import numpy as np

def imShow(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def toCanny(img):
    gaus = cv2.GaussianBlur(img, (5,5), 0)
    canny = cv2.Canny(gaus, 100, 200)
    return canny 

def contours(img, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    max_len_index = 0
    max_len = 0
    for counter, cont in enumerate(contours):
        print (cont)
        if len(cont) > max_len:
            max_len_index = counter
    # print("contours", contours)
    # print('max len index' , max_len_index)
    try:
        x, y, w, h = cv2.boundingRect(contours[max_len_index])
        #relative to shape(shape[1]* .2) <-- sumthin
        print (w, mask.shape)
        if h > w*2 and w > int(img.shape[1] * .2):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            return img, x, y, w, h, True
    
    except: 
        return 'there\'s no cone'

def regionOfInterest(img):
    # 
    bounds = np.int32([[[img.shape[1]* .35, 0], [img.shape[1]* .65, 0], [img.shape[1]* .65, img.shape[0]], [img.shape[1]*.35, img.shape[0]]]])
    mask = np.zeros_like(img)
    #highlight the area
    cv2.fillPoly(mask, bounds, (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    imShow('done', masked_image)
    return masked_image


# insert cone image
orig_image = cv2.imread('coner.jpg')
orig_image = np.copy(orig_image)

#filter out colors we need
low = np.uint8([16, 89, 230])
high = np.uint8([80, 203, 255])
filtered = cv2.inRange(orig_image, low, high)

#filter in area we want 
cam_area = regionOfInterest(filtered)
cannied = toCanny(cam_area)
imShow('cannied', cannied)
try:
    cone, x, y, w, h, isThereCone = contours(orig_image, cam_area)
    cv2.putText(cone, 'cone is here', (((x + w) // 2) + 20, y +  20), cv2.FONT_HERSHEY_PLAIN, int(orig_image.shape[1]*.002), (0, 255, 0), int(orig_image.shape[1] * 0.0005))
    imShow('completed' ,cone)
except:
    pass





