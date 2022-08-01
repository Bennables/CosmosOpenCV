import cv2 as cv2  # importing the library
import numpy as np


def show_image(name: str, image):
    cv2.imshow(name, image)
    cv2.waitKey(20)
    cv2.destroyAllWindows()


def find_canny(img, thresh_low, thresh_high):
    """converts any image into a Canny - params(original image, low threshold, high threshold"""
    # first to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image("gray", img_gray)
    # then to gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    show_image('blur', img_blur)
    # Canny!
    img_canny = cv2.Canny(img_blur, 100, 200)
    show_image('canny', img_canny)
    return img_canny


def region_of_interest(image, bounds):  # function for extracting region of interest
    # bounds in (x,y) format
    # bounds = np.array([[[0,250],[0,200],[150,100],[500,100],[650,200],[650,250]]],dtype=np.int32)
    # creates a polygon
    # notation of coords :
    # bounds = np.array([[[0, 250], [0, 200], [150, 100], [500, 100], [650, 200], [650, 250]]], dtype=np.int32)
    mask = np.zeros_like(image)

    # highlights the area
    # fillpoly takes 3 params: (numpy array(img), bounds(set of points(nparray)), color(b/w 0-255, 255 is white)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    masked_image = cv2.bitwise_and(image, mask)
    show_image('mask', mask)
    return masked_image


def draw_lines(img, lines):  # function for drawing lines on black mask
    mask_lines = np.zeros_like(img)
    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv2.line(mask_lines, (x1, y1), (x2, y2), [0, 0, 255], 2)
    #              image auto hough lines plotted, color,    thickness
    return mask_lines


# plots the image on several points for fixed coordinates.


def get_coordinates(img, line_parameters):  # functions for getting final coordinates
    slope = line_parameters[0]
    intercept = line_parameters[1]
    # y1 = 300
    # y2 =120
    y1 = img.shape[0]
    y2 = 0.6 * img.shape[0]  # takes the middle 60% of th shape.
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, int(y1), x2, int(y2)]


#
# This now creates the path based on the hough lines. there's no origin in the image
# so they try to plot the average line in between the two hough lines.


def compute_average_lines(img, lines):
    left_lane_lines = []
    right_lane_lines = []
    left_weights = []
    right_weights = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        # if x1 == x2, then the slope is undefined, doesn't work ig
        if x2 == x1:
            continue
        # polyfit takes first vertice, second vertice, and degree of the polynomial.
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # implementing polyfit to identify slope and intercept
        slope, intercept = parameters
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if slope < 0:
            # because axes are switched, if y is neg, it will scoot left, if y is pos, it will scoot right
            left_lane_lines.append([slope, intercept])
            left_weights.append(length)
        else:
            right_lane_lines.append([slope, intercept])
            right_weights.append(length)
    # Computing average slope and intercept
    left_average_line = np.average(left_lane_lines, axis=0)
    right_average_line = np.average(right_lane_lines, axis=0)
    # this takes the average of the slope of the line, and the average y intercept.
    print(left_average_line)
    print(right_average_line)
    print(left_average_line, right_average_line)
    # #Computing weigthed sum
    # if len(left_weights)>0:
    #     left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    # #                        np.dot assigns the weight to the line coords. It's the weight array x the coord arrays.
    # if len(right_weights)>0:
    #     right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)
    left_fit_points = get_coordinates(img, left_average_line)
    right_fit_points = get_coordinates(img, right_average_line)

    print(left_fit_points, right_fit_points)
    return [[left_fit_points], [right_fit_points]]  # returning the final coordinates

# image mapping w/ hough lines and gradient changes
# image1 = cv2.imread('test_image.png')
# lane_image = np.copy(image1)
# lane_canny = find_canny(image1, 100, 200)
# lane_roi = region_of_interest(lane_canny, np.array([[[0, image.shape[0]], [0, image.shape[0] / 2], [900, image.shape[0]/2], [900, image.shape[0]]]],
#                     dtype=np.int32))
# lane_lines = cv2.HoughLinesP(lane_roi, 2, np.pi / 180, 50, 40, 5)
# print(lane_lines)
# lane_lines_plotted = draw_lines(lane_image, lane_lines)
# show_image('lines', lane_lines_plotted)
# result_lines = compute_average_lines(lane_image, lane_lines)
# final_lines_mask = draw_lines(lane_image, result_lines)
# show_image('final', final_lines_mask)
#
# #Plotting the final lines on main image
# for points in result_lines:
#     x1,y1,x2,y2 = points[0]
#     cv2.line(lane_image,(x1,y1),(x2,y2),(0,0,255),2)
#
# show_image('yesh', lane_image)


#Video Processing:
# Opens the video
cap = cv2.VideoCapture("test1.mp4")

if not cap.isOpened:
    print('Error opening video capture')
    exit(0)
cap.set(5, 20)
# reads it
while True:
    ret, frame = cap.read()
    if frame is None:
        print(' No captured frame -- Break!')
        break
    lane_image_2 = np.copy(frame)
    lane_image_2 =cv2.cvtColor(lane_image_2,cv2.COLOR_BGR2HLS)
    # show_image('hsv_input',lane_image_2)
    lower_white_hls = np.uint8([  0, 200,   0])
    upper_white_hls = np.uint8([255, 255, 255])
    lane_white_mask = cv2.inRange(lane_image_2,lower_white_hls,upper_white_hls)
    # show_image('whitemask',lane_white_mask)
    lane_image_mask = cv2.bitwise_and(lane_image_2,lane_image_2,mask=lane_white_mask)
    # show_image('bitmask',lane_image_mask)
    lane_canny_2 = find_canny(lane_image_mask,50,150)
    lane_roi_2 = region_of_interest(lane_canny_2, np.array([[[0, image.shape[0]], [0, image.shape[0] / 2], [900, image.shape[0]/2], [900, image.shape[0]]]],
                      dtype=np.int32))
    lane_lines_2 = cv2.HoughLinesP(lane_roi_2,1,np.pi/180,15,5,15)
    lane_lines_plotted_2 = draw_lines(lane_image_2,lane_lines_2)
    result_lines_2 = compute_average_lines(lane_image_2,lane_lines_2)
    final_lines_mask_2 = draw_lines(lane_image_2,result_lines_2)
    # show_image('final',final_lines_mask_2)

    for points in result_lines_2:
        x1,y1,x2,y2 = points[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

    show_image('output',frame)

cap = cv2.VideoCapture(0) # opening the camera

if not cap.isOpened:
    print('Error opening video capture')
    exit(0)
cap.set(5,60)

while True:
    ret, frame = cap.read() # reading the frame


    edges = find_canny(frame, 100, 200, (5, 5)) # get edges
    cv2.imshow('frame', edges) # displaying the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# inefficient code, added into function
# # reads the original image and displays
# image = cv2.imread('track_image.jpg')
# show_image('fig1', image)
#
# # creates a copy to convert to gray.
# lane_image = np.copy(image)
# lane_gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow('gray', lane_gray)
# show_image('fig2', lane_gray)
#
# # canny filter looks for large changes
#
# # Gaussian Blue: params(image,(kernel size), kernel stdev on x)
# lane_blur = cv2.GaussianBlur(lane_gray, (5, 5), 0)
#
# # apply canny: params(gaussian blurred, lower threshold, upper threshold)
# # you are looking for changes between 100 and 200 below
# lane_canny = cv2.Canny(lane_blur, 100, 200)
# show_image('fig3', lane_canny)
