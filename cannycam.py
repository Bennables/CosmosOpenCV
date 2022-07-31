import cv2 as cv2  # importing the library
import numpy as np  # importing numpy library


def show_image(name: str, image):
    cv2.imshow(name, image)
    cv2.waitKey(20)
    cv2.destroyAllWindows()


def findCanny(image, lower_thresh, upper_thresh, kernel_size):  # get edges
    copiedImage = np.copy(image)
    grayImage = cv2.cvtColor(copiedImage, cv2.COLOR_BGR2GRAY )  # grayscale
    blurredImage = cv2.GaussianBlur(grayImage, kernel_size, 0)  # gaussian blur for smoothing
    cannyImage = cv2.Canny(blurredImage, lower_thresh, upper_thresh)  # canny edge detection
    return cannyImage


def regionOfInterest(image):  # function for extracting region of interest
    # bounds in (x,y) format
    # bounds = np.array([[[0,250],[0,200],[150,100],[500,100],[650,200],[650,250]]],dtype=np.int32)
    # creates a polygon
    # notation of coords :
    # bounds = np.array([[[0, 250], [0, 200], [150, 100], [500, 100], [650, 200], [650, 250]]], dtype=np.int32)
    bounds = np.array([[[0, image.shape[0]], [0, image.shape[0] / 2], [900, image.shape[0]/2], [900, image.shape[0]]]],
                      dtype=np.int32)
    mask = np.zeros_like(image)

    # highlights the area
    # fillpoly takes 3 params: (numpy array(img), bounds(set of points(nparray)), color(b/w 0-255, 255 is white)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    masked_image = cv2.bitwise_and(image, mask)
    show_image('mask', mask)
    return masked_image


def findLines(image):  # find lines with hough transform
    cannyImage = findCanny(image, 100, 200, (5, 5))
    croppedImage = regionOfInterest(cannyImage)
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    return lines


def displayLines(image, lines):  # display lines
    line_image = np.zeros_like(image)  # blank image in shape of image
    if lines is not None:  # check if lines is empty
        for line in lines:  # loop through lines
            x1, y1, x2, y2 = line[0]  # get coordinates of lines
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # plot each line
    return line_image


cap = cv2.VideoCapture(0)  # opening the camera

if not cap.isOpened:
    print('Error opening video capture')
    exit(0)
cap.set(5, 60)

while True:
    ret, frame = cap.read()  # reading the frame

    edges = findCanny(frame, 100, 200, (5, 5))  # get edges
    cv2.imshow('frame', edges)  # displaying the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
