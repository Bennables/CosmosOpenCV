import math
from turtle import right
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

#images for np are height x width

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
    """paints the canny lines in the roi. image is already cannyd,
    so it uses bitwise and to show just the lines in the roi"""
    # starting x, ending x, starting y, ending y
    # shape is (height, width, depth)
    # the bounds are shape doesn't matter as long as you go in one direction, mix and you get triangle.
    bounds = np.array([[[0, 0], [0,image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1],0]]], dtype=np.int32)
    #sets everything to zeros
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255,0,0])
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
    
def draw_lines(img,lines):
    # creates an array with 0s same shape as img
    # plots points.
    print('lines: ', lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
    return img


def get_coordinates(img, line_parameters):
    #print('line_parameters: ', line_parameters)
    #print(type(line_parameters))
    if type(line_parameters) is np.ndarray:
        slope = line_parameters[0]
        intercept = line_parameters[1]
        y1 = int(img.shape[0])
        y2 = int(.6 * img.shape[0])
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]

def computeAverageLines(image, lines): # smooth lines by averaging them
    leftLaneLines = []
    rightLaneLines = []
    leftWeights = []
    rightWeights = []

    for points in lines:
        # print(points)
        x1, y1, x2, y2 = points[0] # get coordinates of lines

        if x1 == x2:
            continue

        parameters = np.polyfit((x1, x2), (y1, y2), 1) # polynomial fit to identify slope and intercept
        slope, intercept = parameters
        length = np.sqrt((y2 - y1)** 2 + (x2 - x1)**2)

        if slope < 0: # y axis decreases upwards
            leftLaneLines.append([slope, intercept]) 
            leftWeights.append(length)

        else: # y axis increases downwards
            rightLaneLines.append([slope, intercept])
            rightWeights.append(length)

    leftAverageLine = np.average(leftLaneLines, axis=0)
    rightAverageLine = np.average(rightLaneLines, axis=0)
    #print(leftAverageLine)
    #print(rightAverageLine)
    if type(leftAverageLine) is not np.ndarray:
        return None

    if type(rightAverageLine) is not np.ndarray:
        return None

    leftFitPoints = get_coordinates(image, leftAverageLine)
    rightFitPoints = get_coordinates(image, rightAverageLine)
    # print(leftFitPoints)
    # print(rightFitPoints)
    return [[leftFitPoints], [rightFitPoints]]
    
    
# image mapping w/ hough lines and gradient changes
image1 = cv2.imread('stop.jpg')
lane_image = np.copy(image1)
lane_canny = find_canny(image1, 100, 200)
#show_image('hi', lane_canny)

lane_roi = region_of_interest(lane_canny)
lane_lines = cv2.HoughLinesP(lane_roi, 1, math.pi/180, 15, minLineLength=10, maxLineGap=10)
    #lane_roi, 3, 5, np.pi / 180, 50, 40, 5)
lane_lines_plotted = draw_lines(lane_image, lane_lines)
show_image('lines', lane_lines_plotted)
result_lines = computeAverageLines(lane_image, lane_lines)
print('result lines' ,result_lines)
if result_lines is not None:
    print('result_lines type', type(result_lines))
    print('######################################')
    final_lines_mask = draw_lines(lane_image, result_lines)

    #Plotting the final lines on main image
    for points in result_lines:
        x1,y1,x2,y2 = points[0]
        cv2.line(lane_image,(x1,y1),(x2,y2),(0,0,255),2)

    show_image('yesh', lane_image)
    print ('kdfjlds')