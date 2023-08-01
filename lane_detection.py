import cv2
import numpy as np
import matplotlib.pyplot as plt
I = cv2.imread('test_image4.png')
cap = cv2.VideoCapture('AUV_Vid.mkv')
from random import randrange

def detect_lines(img, threshold1 = 50, threshold2 = 150, apertureSize = 3, minLineLength = 100, maxLineGap = 10):
    
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # grayCon = cv2.addWeighted(gray, 2, gray, 0, 0)
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize) # detect edges
    lines = cv2.HoughLinesP(
                    edges,
                    1,
                    np.pi/180,
                    100,
                    minLineLength=minLineLength,
                    maxLineGap=maxLineGap,
            ) # detect lines

    lineList = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            linexy = [x1, y1, x2, y2]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lineList.append(linexy)
    return lineList

def draw_lines(img, lines, color = (0, 255, 0)):
   
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, 3) 
    return img

def get_slopes_intercepts(img, lines):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    image = img
    slopes = []
    xIntercepts = []
    height = image.shape[0]
    for line in lines:
        if line[0] == line[2]:
            line[0] += 1
        slope = (line[1] - line[3]) / (line[0] - line[2])
        slopes.append(slope)
        xIntercepts.append((height-line[1])/slope + line[0])
    return slopes, xIntercepts

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









