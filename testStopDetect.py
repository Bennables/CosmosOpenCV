"""
To-Do
create object to let it stop/continue
make functions for stopping for fixed time, and continuing
clean up code
create boolean for if there's a stop sign
comment for clarity
"""

from tracemalloc import stop
import cv2 as cv2
from cv2 import imshow
from cv2 import findContours
from cv2 import drawContours
from matplotlib import image
from matplotlib.pyplot import gray
import depthai as dai
import numpy as np
import time
from functionsForLaneDetection import VESC


#shows image, 
def imShow(img, name = 'image'):
    """Shows the img(np.array). click any key to close image or change waitKey from 0 to somethign else, time = ms
    Params: img(np.array), name(string)<optional>"""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#outlines the objects in the image by detecting color gradient changes    
def toCanny(img):
    """outlines objects in the image, returns canny
    Prams: img(np.array)"""
    gray_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus_x = cv2.GaussianBlur(gray_x, (5,5), 0)
    canny_x = cv2.Canny(gaus_x, 100, 200)
    return canny_x


#finds the contours(outlines the items of shame shade, binary mask is best)
def findcontours(img, mask):
    """Finds the contours of the mask and places them on img. returns img with contours on it
    Params: img(np.array), mask(can be rgb image, easier to do black and white)
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(img, contours, -1, (0,255,0), 3)
    max_len_index = 0
    max_len = 0
    for counter, cont in enumerate(contours):
        if len(cont) > max_len:
            print(len(cont), counter)
            max_len = len(cont)
            max_len_index = counter
    print('max len index' , max_len_index)
    # try:
        
    x, y, w, h = cv2.boundingRect(contours[max_len_index])
    #relative to shape(shape[1]* .2) <-- sumthin
    if w > int(img.shape[1] * .2):
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        return img, x, y, w, h, True
    # except:
    #     print('img not found')


# creating vesc object
car = VESC(serial_port='/dev/ttyACM0')
# Video Processing:
# sets up camera and sends it
pipeline = dai.Pipeline()
# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(600, 600)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)
count=1
# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
# Start pipeline
    device.startPipeline()
    controlQueue = device.getInputQueue('control')
    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        car.run(.5, .2)
        time.sleep(3)
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        controlQueue.send(ctrl)
        lane_image = inRgb.getCvFrame()
        #unecessary
        orig_image = lane_image.copy()

        #changes the image to hsv color space and makes masks for red, tuples are bgr values
        frame_hsv = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(frame_hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(frame_hsv, (120,50,20), (180,255,255))
        #combines the 2 masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        try:
            #getting contours for the image and getting dimensions, also bool for it there is stop sign identified
            stopSign, x, y, w, h, isStopSign = findcontours(orig_image, mask)
            #adding text to the image
            cv2.putText(stopSign, 'Stop sign detected!', (((x + w) // 2) - 10, y - 20), cv2.FONT_HERSHEY_PLAIN, int(orig_image.shape[1]*.002), (0, 255, 0), int(orig_image.shape[1] * 0.0005))
        except:
            isStopSign = False
            pass

        #if there's a stop sign, it will stop.
        if isStopSign:
            print('it worked')
            car.run(0.5, 0)
            time.sleep(1)
            isStopSign = False

        else:
            print('fail')
        

            






