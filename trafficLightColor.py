import numpy as np
import cv2
import matplotlib.pyplot as plt

def estimate_label(rgb_image): # Standardized RGB image
    return red_green_yellow(rgb_image)

def findNonZero(rgb_image):
    rows, cols, _ = rgb_image.shape
    counter = 0

    for row in range(rows):
      for col in range(cols):
        pixel = rgb_image[row, col]
        if sum(pixel) != 0:
          counter = counter + 1

    return counter

def red_green_yellow(rgb_image):
    '''Determines the Red, Green, and Yellow content in each image using HSV and
    experimentally determined thresholds. Returns a classification based on the
    values.
    '''
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # Green
    lower_green = np.array([55, 40, 40])
    upper_green = np.array([90, 255,255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_result = cv2.bitwise_and(rgb_image, rgb_image, mask = green_mask)

    # Yellow
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([32, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask = yellow_mask)

    # Red
    lower_red = np.array([0,50,20])
    upper_red = np.array([5,255,255])
    red_mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([175,50,20])
    upper_red = np.array([180,255,255])
    red_mask2 = cv2.inRange(hsv, lower_red, upper_red)

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask = red_mask)

    sum_green = findNonZero(green_result)
    sum_yellow = findNonZero(yellow_result)
    sum_red = findNonZero(red_result)
    print('\nred: ', sum_red)
    print('yellow: ', sum_yellow)
    print('green: ', sum_green)
    height, width, _ = green_result.shape
    number_pixel = height*width
    print(number_pixel)
    if sum_red < 0.1*number_pixel and sum_yellow < 0.01*number_pixel and sum_green < 0.06*number_pixel:
      return "black"
    elif sum_red >= sum_yellow and sum_red >= sum_green:
      return "red"
    elif sum_yellow >= sum_green and sum_yellow >= sum_red:
      return "yellow"
    elif sum_green >= sum_yellow and sum_green >= sum_red:
      return "green"