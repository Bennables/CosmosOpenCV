

import cv2 # Import the OpenCV library to enable computer vision

import numpy as np # Import the NumPy scientific computing library

import edge_detection as edge # Handles the detection of lane lines

import matplotlib.pyplot as plt # Used for plotting and error checking

import pyvesc as pyvesc

import time

 

class VESC:

  '''

  VESC Motor controler using pyvesc

  This is used for most electric skateboards.

  inputs: serial_port---- port used communicate with vesc. for linux should be something like /dev/ttyACM1

  has_sensor=False------- default value from pyvesc

  start_heartbeat=True----default value from pyvesc (I believe this sets up a heartbeat and kills speed if lost)

  baudrate=115200--------- baudrate used for communication with VESC

  timeout=0.05-------------time it will try before giving up on establishing connection

  percent=.2--------------max percentage of the dutycycle that the motor will be set to

  outputs: none

  uses the pyvesc library to open communication with the VESC and sets the servo to the angle (0-1) and the duty_cycle(speed of the car) to the throttle (mapped so that percentage will be max/min speed)

  Note that this depends on pyvesc, but using pip install pyvesc will create a pyvesc file that

  can only set the speed, but not set the servo angle.

  Instead please use:

  pip install git+https://github.com/LiamBindle/PyVESC.git@master

  to install the pyvesc library

  '''

  def __init__(self, serial_port, percent=.2, has_sensor=False, start_heartbeat=True, baudrate=115200, timeout=0.05, steering_scale = 1.0, steering_offset = 0.0 ):

    try:

      import pyvesc

    except Exception as err:

      print("\n\n\n\n", err, "\n")

      print("please use the following command to import pyvesc so that you can also set")

      print("the servo position:")

      print("pip install git+https://github.com/LiamBindle/PyVESC.git@master")

      print("\n\n\n")

      time.sleep(1)

      raise

    assert percent <= 1 and percent >= -1,'\n\nOnly percentages are allowed for MAX_VESC_SPEED (we recommend a value of about .2) (negative values flip direction of motor)'

    self.steering_scale = steering_scale

    self.steering_offset = steering_offset

    self.percent = percent

    try:

      self.v = pyvesc.VESC(serial_port, has_sensor, start_heartbeat, baudrate, timeout)

    except Exception as err:

      print("\n\n\n\n", err)

      print("\n\nto fix permission denied errors, try running the following command:")

      print("sudo chmod a+rw {}".format(serial_port), "\n\n\n\n")

      time.sleep(1)

      raise

  def run(self, angle, throttle):

    self.v.set_servo((angle * self.steering_scale) + self.steering_offset)

    self.v.set_duty_cycle(throttle*self.percent)

 

  def stop(self, angle, throttle):

    self.v.set_servo((0) + 0)

    self.v.set_duty_cycle(0)

 

class Lane:

  """

  Represents a lane on a road.

  """

  def __init__(self, orig_frame):

    """

    Default constructor

   

    :param orig_frame: Original camera image (i.e. frame)

    """

    self.orig_frame = orig_frame

 

    # This will hold an image with the lane lines  

    self.lane_line_markings = None

 

    # This will hold the image after perspective transformation

    self.warped_frame = None

    self.transformation_matrix = None

    self.inv_transformation_matrix = None

 

    # (Width, Height) of the original video frame (or image)

    self.orig_image_size = self.orig_frame.shape[::-1][1:]

 

    width = self.orig_image_size[0]

    height = self.orig_image_size[1]

    self.width = width

    self.height = height

 

    # Four corners of the trapezoid-shaped region of interest

    # You need to find these corners manually.

    self.roi_points = np.float32([

      (int(0.456*width),int(0.544*height)), # Top-left corner

      (0, height-1), # Bottom-left corner    

      (int(0.958*width),height-1), # Bottom-right corner

      (int(0.6183*width),int(0.544*height)) # Top-right corner

    ])

   

    # The desired corner locations  of the region of interest

    # after we perform perspective transformation.

    # Assume image width of 600, padding == 150.

    self.padding = int(0.25 * width) # padding from side of the image in pixels

    self.desired_roi_points = np.float32([

      [self.padding, 0], # Top-left corner

      [self.padding, self.orig_image_size[1]], # Bottom-left corner    

      [self.orig_image_size[

        0]-self.padding, self.orig_image_size[1]], # Bottom-right corner

      [self.orig_image_size[0]-self.padding, 0] # Top-right corner

    ])

   

    # Histogram that shows the white pixel peaks for lane line detection

    self.histogram = None

   

    # Sliding window parameters

    self.no_of_windows = 10

    self.margin = int((1/12) * width)  # Window width is +/- margin

    self.minpix = int((1/24) * width)  # Min no. of pixels to recenter window

   

    # Best fit polynomial lines for left line and right line of the lane

    self.left_fit = None

    self.right_fit = None

    self.left_lane_inds = None

    self.right_lane_inds = None

    self.ploty = None

    self.left_fitx = None

    self.right_fitx = None

    self.leftx = None

    self.rightx = None

    self.lefty = None

    self.righty = None

   

    # Pixel parameters for x and y dimensions

    self.YM_PER_PIX = 7.0 / 400 # meters per pixel in y dimension

    self.XM_PER_PIX = 3.7 / 255 # meters per pixel in x dimension

   

    # Radii of curvature and offset

    self.left_curvem = None

    self.right_curvem = None

    self.center_offset = None

 

  def calculate_car_position(self, print_to_terminal=False):

    """

    Calculate the position of the car relative to the center

   

    :param: print_to_terminal Display data to console if True  

    :return: Offset from the center of the lane

    """

    # Assume the camera is centered in the image.

    # Get position of car in centimeters

    car_location = self.orig_frame.shape[1] / 2

 

    # Fine the x coordinate of the lane line bottom

    height = self.orig_frame.shape[0]

    bottom_left = self.left_fit[0]*height**2 + self.left_fit[

      1]*height + self.left_fit[2]

    bottom_right = self.right_fit[0]*height**2 + self.right_fit[

      1]*height + self.right_fit[2]

 

    center_lane = (bottom_right - bottom_left)/2 + bottom_left

    center_offset = (np.abs(car_location) - np.abs(

      center_lane)) * self.XM_PER_PIX * 100

 

    if print_to_terminal == True:

      print(str(center_offset) + 'cm')

     

    self.center_offset = center_offset

     

    return center_offset

 

  def calculate_curvature(self, print_to_terminal=False):

    """

    Calculate the road curvature in meters.

    :param: print_to_terminal Display data to console if True

    :return: Radii of curvature

    """

    # Set the y-value where we want to calculate the road curvature.

    # Select the maximum y-value, which is the bottom of the frame.

    y_eval = np.max