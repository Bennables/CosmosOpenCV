"""
change direction
comment it up
"""

import cv2 as cv2
import numpy as np
from lane_functions import VESC
import depthai as dai
import time

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
    #highlight the are
    cv2.fillPoly(mask, bounds, (255,255,255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# # insert cone image
# orig_image = cv2.imread('coner.jpg')
# orig_image = np.copy(orig_image)

# #filter out colors we need
# # low = np.uint8([16, 89, 230])
# # high = np.uint8([80, 203, 255])
# # filtered = cv2.inRange(orig_image, low, high)
# frame_hsv = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV) 
# mask = cv2.inRange(frame_hsv, (10, 100, 20), (25, 255, 255))
# #filter in area we want 
# cam_area = regionOfInterest(mask)
# cannied = toCanny(cam_area)
# try:
#     cone, x, y, w, h, isThereCone = contours(orig_image, cam_area)
#     cv2.putText(cone, 'cone is here', (((x + w) // 2) + 20, y +  20), cv2.FONT_HERSHEY_PLAIN, int(orig_image.shape[1]*.002), (0, 255, 0), int(orig_image.shape[1] * 0.0005))
#     imShow('completed' ,cone)
# except:
#     pass





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
        orig_image = np.copy(lane_image)


        #imShow('canny', cannied) 
        frame_hsv = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(frame_hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(frame_hsv, (120,50,20), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # low = np.uint8([0, 0, 100])
        # high = np.uint8([30, 30, 255])
        # mask = cv2.inRange(orig_image, low, high)
        # imShow('mask', mask)
        cam_area = regionOfInterest(mask)
        try:
            cone, x, y, w, h, isCone = contours(orig_image, cam_area)
            cv2.putText(cone, 'cone is here', (((x + w) // 2) + 20, y +  20), cv2.FONT_HERSHEY_PLAIN, int(orig_image.shape[1]*.002), (0, 255, 0), int(orig_image.shape[1] * 0.0005))
            imShow('hi', cone)
        except:
            isCone = False
            pass

        if isCone:
            car.run(.75, .2)
            time.sleep(1.5)
            car.run(.1, .2)
            time.sleep(1.5)
            car.run(.6, .2)
            time.sleep(1.5)
            car.run(.5, .2)
            time.sleep(3)
            isCone = False
            