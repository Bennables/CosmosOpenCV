from asyncio import as_completed
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(name, img):
    """ params: box title, imread image
    Show the image you put in"""
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_canny(img, lower_threshold, upper_threshold):
    """ takes imread image, lower threshold, upper threshold for contrast"""
    img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Gaussian blur takes the image, kernel size, and kernal stdev, also takes more 
    #look thru documentation
    img_gaussian_blur = cv2.GaussianBlur(img_grayscale, (5,5), 0)
    #takes the gaussian image, lower threshold, and upper for gradient changes.
    img_canny = cv2.Canny(img_gaussian_blur, lower_threshold, upper_threshold)
    return img_canny

def region_of_interest(image):
    """Extracts the region of interest, bounds are in x,y format"""
    bounds = np.array([[[]]])
    mask = np.zeros_like(image)
    
