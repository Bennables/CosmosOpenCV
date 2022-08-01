from operator import inv
from tkinter import Frame
import cv2 as cv2
from cv2 import drawContours
from cv2 import sort #importing the library
import numpy as np #importing numpy library
import matplotlib.pyplot as plt
import math
import time

REGION_Y_LOWER = 460
REGION_Y_UPPER = 300
REGION_X_LOWER_LEFT = 100
REGION_X_UPPER_LEFT = 350
REGION_X_UPPER_RIGHT = 590
REGION_X_LOWER_RIGHT = 630

def findCanny(image, lower_thresh, upper_thresh, kernel_size): # get edges
    copiedImage = np.copy(image)
    grayImage = cv2.cvtColor(copiedImage, cv2.COLOR_BGR2GRAY) # grayscale
    blurredImage = cv2.GaussianBlur(grayImage, kernel_size, 0) # gaussian blur for smoothing
    cannyImage = cv2.Canny(blurredImage, lower_thresh, upper_thresh) # canny edge detection
    #cv2.imshow('canny', cannyImage)
    return cannyImage

def regionOfInterest(image): # define region of interest
    polygon = np.array([[[REGION_X_LOWER_LEFT, REGION_Y_LOWER],  [REGION_X_LOWER_RIGHT, REGION_Y_LOWER], [REGION_X_UPPER_RIGHT, REGION_Y_UPPER], [REGION_X_UPPER_LEFT, REGION_Y_UPPER]]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage
    

def findLines(image): # find lines with hough transform
    cannyImage = findCanny(image, 50, 200, (5, 5))
    croppedImage = regionOfInterest(cannyImage)
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 50, np.array([]), minLineLength=25, maxLineGap=8)
    
    return lines

def displayLines(image, lines): # display lines
    line_image = np.zeros_like(image) # blank image in shape of image

    xVals = []
    yVals = []

    
    if lines is not None: # check if lines is empty
        for line in lines: # loop through lines
            #print(line)
            x1, y1, x2, y2 = line[0] # get coordinates of lines
            xVals.append(x1)
            yVals.append(y1)
            xVals.append(x2)
            yVals.append(y2)
    points = []

    while(len(points) < 2):
        ind = yVals.index(min(yVals))
        if(len(points) == 0):
            points.append([xVals[ind], yVals[ind]])
            xVals.pop(ind)
            yVals.pop(ind)
        if(len(points) == 1):
            ind = yVals.index(min(yVals))
            while(xVals[ind] in range (points[0][0] - 100, points[0][0] + 100) or yVals[ind] not in range (points[0][1] - 100, points[0][1] + 100)):
                xVals.pop(ind)
                yVals.pop(ind)
                ind = yVals.index(min(yVals))
            points.append([xVals[ind], yVals[ind]])
        
        ind = yVals.index(max(yVals))
        if(len(points) == 2):
            ind = yVals.index(max(yVals))
            while(xVals[ind] not in range (points[1][0] - 250, points[1][0] + 250)):
                xVals.pop(ind)
                yVals.pop(ind)
                ind = yVals.index(max(yVals))
            points.append([xVals[ind], yVals[ind]])
        if(len(points) == 3):
            ind = yVals.index(max(yVals))
            while(xVals[ind] in range (points[2][0] - 100, points[2][0] + 100)):
                xVals.pop(ind)
                yVals.pop(ind)
                ind = yVals.index(max(yVals))
            points.append([xVals[ind], yVals[ind]])
        
        #print(points)
        #cv2.waitKey(0)
    
    cv2.fillPoly(image, np.int32([points]), color=(0, 255, 0))
    
    return image

def getCoordinates(image, lineParameters):
    if type(lineParameters) is np.ndarray:
        slope = lineParameters[0]
        intercept = lineParameters[1]

        y1 = image.shape[0]
        y2 = image.shape[0] * 0.7

        x1 = int((y1 - intercept)/ slope)
        x2 = int((y2 - intercept)/ slope)

        return [int(x1), int(y1), int(x2), int(y2)]

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
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

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
    
    leftFitPoints = getCoordinates(image, leftAverageLine)
    rightFitPoints = getCoordinates(image, rightAverageLine)
    # print(leftFitPoints)
    # print(rightFitPoints)
    return [[leftFitPoints], [rightFitPoints]]

def rescaleFrame(frame, percent):
    scale_percent = percent
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


def curves(image, mask):

    allPoints = np.ndarray(shape=(0, 2))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    

    for cont in contours:
        for point in cont:
            allPoints = np.append(allPoints, point, axis=0)
    sortedPoints = allPoints[allPoints[:,1].argsort()]
    
    finalPoints = []
    finalPoints.append((int(sortedPoints[0][0].item()),int(sortedPoints[0][1].item())))

   
    for point in sortedPoints:
        if int(point[0].item()) in range(finalPoints[0][0] - 80, finalPoints[0][0] + 80):
            continue
        else:
            finalPoints.append((int(point[0].item()),int(point[1].item())))
            break
        
    sortedPoints = allPoints[allPoints[:,1].argsort()]
    sortedPoints = np.flip(sortedPoints, 0)
    #print(sortedPoints)

    
    finalPoints.append((int(sortedPoints[0][0].item()),int(sortedPoints[0][1].item())))

    
    for point in sortedPoints:
        if(len(finalPoints) == 2):
            if finalPoints[-1][0] <= (image.shape[1] // 2):
                if int(point[0].item()) < (image.shape[1] // 2):
                    finalPoints.append((int(point[0].item()),int(point[1].item())))
                    break
            else:
                if int(point[0].item()) > (image.shape[1] // 2):
                    finalPoints.append((int(point[0].item()),int(point[1].item())))
                    break
        
    for point in sortedPoints:
        if(len(finalPoints) == 3):
            if finalPoints[-1][0] <= (image.shape[1] // 2):
                if int(point[0].item()) > (image.shape[1] // 2):
                    finalPoints.append((int(point[0].item()),int(point[1].item())))
                    continue
            else:
                if int(point[0].item()) < (image.shape[0] // 2):
                    finalPoints.append((int(point[0].item()),int(point[1].item())))
                    continue

        
    #print(sortedPoints)
    


    
    finalPoints = sorted(finalPoints, key = lambda x: x[0])
    if finalPoints[2][1] > finalPoints[3][1]:
        print('swapped!')
        finalPoints[2], finalPoints[3] = finalPoints[3], finalPoints[2]

    print(finalPoints)

    cv2.fillPoly(image, np.int32([finalPoints]), color=(0, 255, 0))

    

    return image