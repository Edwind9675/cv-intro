import cv2
import numpy as np
import matplotlib.pyplot as plt
I = cv2.imread('test_image4.png')
cap = cv2.VideoCapture('AUV_Vid.mkv')
from random import randrange
# from dt_apriltags import Detector
"""
def image_line_detect(img0):
  
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) # convert to grayscale
    edges = cv2.Canny(gray, 470, 301, apertureSize=5) # detect edges
    lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi/180,
                100,
                minLineLength=300,
                maxLineGap=10,
        ) # detect lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        slope = ((y2-y1)/(x2-x1))
        print (lines)

    #image_line_detect(img)
    plt.imshow(img0)

def draw_lines (file_name, coors):
    #linescoor = float(input(e))
    pic = cv2.imread(file_name)
    coors = ()
    x = coors[0::2] 
    y = coors[1::2]
    cv2.line(pic, (x.x1, y.y1), (x.x2, y.y2), (0, 255, 0), 2)


def get_slopes_intercepts(file_name, lines):
    pic = cv2.imread(file_name)
    lines = []
    xint = []
    for i in lines:
        x1, y1, x2, y2 = lines[i]
        cv2.line(pic, (x1, y1), (x2, y2), (0, 255, 0), 2)
        slope = ((y2-y1)/(x2-x1))
        print (lines)
        if slope [i]*1.04 == slope[i+1]:
            return slope[i]


    

    #image_line_detect(img)
    plt.imshow(slope[i])
print ()
def get_lane_center(lines):
    for i in la
    if lines[i] = -lines[i+1]
return 

def detect_lanes(lines): 

    """
#if 2 lanes are opposite direction, that mean they are opposite

def image_line_detect(img):
    """detect lines in the area"""
    image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    edges = cv2.Canny(gray, 90, 100, apertureSize=3) # detect edges
    lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi/180,
                100,
                minLineLength=300,
                maxLineGap=10,
        ) # detect lines

    lineList = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        linexy = [x1, y1, x2, y2]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lineList.append(linexy)
        return lineList

    #image_line_detect(img)
    

def draw_lines(img, lines, color = (0, 255, 0)):
    image = cv2.imread(img)
    for line in lines:
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), color, 3) 
    return image

def get_slopes_intercepts(img, lines):
    image = cv2.imread(img)
    slopes = []
    xIntercepts = []
    height = image.shape[0]
    for line in lines:
        slope = (line[1] - line[3]) / (line[0] - line[2])
        slopes.append(slope)
        xIntercepts.append((height-line[1])/slope + line[0])
    return slopes, xIntercepts #slopes is a list of slope

def detect_lanes(img, lines):
    slopes, xIntercepts = get_slopes_intercepts(lines)
    lineDict = dict(zip(xIntercepts, slopes))
    height = cv2.imread(img).shape[0]
    possibleLanes = []
    if len(slopes)> 1:
        for i in range(0,len(slopes)):
            # if (len(slopeList) > 1):
            #     i += 1
            #     print("added i")
            for j in range (i+1,len(slopes)):
                print(f"DistREQ:{abs(xIntercepts[i]-xIntercepts[j])}")
                print(f"slopeREQ:{abs(1/ slopes[i]-1 /slopes[j])}")
                if(abs(xIntercepts[i]-xIntercepts[j])< 10000 and abs(1/ slopes[i]-1 /slopes[j]) < 1):
                    
                    xPoint = ((slopes[i] * xIntercepts[i]) - (slopes[j] * xIntercepts[j]))/(slopes[i]-slopes[j])
                    yPoint = slopes[i]*(xPoint - xIntercepts[i]) + 1080
                    
                    # avgSlope = (slopeList[i]+ slopeList[j])/2
                    # avgInterecept = (xInterceptList[i]+xInterceptList[j])/2
                    lane1 = [xIntercepts[i], 1080, xPoint,yPoint]
                    lane2 = [xIntercepts[j], 1080, xPoint,yPoint]
                    addedlanes = [lane1,lane2]
                    #print (f"thiasdfee:{(slopeList[i] * xInterceptList[i]) - slopeList[j] * xInterceptList[j]}")
                    possibleLanes.append(addedlanes)

    return possibleLanes

def draw_lanes(img, possibleLanes):
    color_of_lanes = (randrange(255), randrange(255), randrange(255))
    for addedlanes in possibleLanes:
        x1, y1, x2, y2 = possibleLanes
        print (possibleLanes)
        cv2.line((x1, y1), (x2, y2), color_of_lanes, 2)
    return img        
