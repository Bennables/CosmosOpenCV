"""
To-Do
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
import depthai as dai
import numpy as np
import time
from lane_functions import VESC
import serial.tools.list_ports
print(list(serial.tools.list_ports.comports()))

#function for showing images using cv2
def imShow(name, img):
    """shows images, params: name(str), img(np.array)"""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#changes image to canny
def toCanny(img):
    """changes image to canny using grayscale-> gaussian -> canny. returns canny
    params: image(np.array)"""
    gray_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus_x = cv2.GaussianBlur(gray_x, (5,5), 0)
    canny_x = cv2.Canny(gaus_x, 100, 200)
    return canny_x

# finds contours and finds the largest to plot a box
def contours(img, mask):
    """finds the contours, finds the largest contour, and then creates rectangle if contour fig is big enough
    params: img to lay over(np.array), mask(binary img preferably for contour scanning)"""
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
        
        return img, x, y, w, h, True
    except:
        print('img not found')


car = VESC(serial_port='COM3')
# Video Processing:
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
        orig_image = cv2.imread('lane_image')
        orig_image = orig_image.copy()
        cannied = toCanny(orig_image)
        #imShow('canny', cannied)

        low = np.uint8([0, 0, 100])
        high = np.uint8([30, 30, 255])
        mask = cv2.inRange(orig_image, low, high)
        imShow('mask', mask)
        try:
            stopSign, x, y, w, h, isStopSign = contours(orig_image, mask)
            cv2.putText(stopSign, 'Stop sign detected!', (((x + w) // 2) - 10, y - 20), cv2.FONT_HERSHEY_PLAIN, int(orig_image.shape[1]*.002), (0, 255, 0), int(orig_image.shape[1] * 0.0005))
        except:
            isStopSign = False
            pass

        if isStopSign:
            car.stop()
            time.sleep(3)
            






