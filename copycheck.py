import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lines(img, threshold1 = 50, threshold2 = 150, apertureSize = 3, minLineLength = 100, maxLineGap = 10):
    """Detects if lines are present in an image (pool).
    args:
        img (image path or np.ndarray): image of pool to detect lines from
        threshold1 (int): lower threshold to detect lines (default 50)
        threshold2 (int): upper threshold to detect lines (default 150)
        apertureSize (odd int): amount of details cv2 will be taking (default 3)
        minLineLength (int): the minimum line length of lines detected (default 100)
        maxLineGap (int): the maximum line gap of lines detected (default 10)
    returns:
        (list): list of points [[x1, y2, x2, y2], ...] that correspond to detected lines
    """
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
    """Returns an image with specified lines drawn on
    args:
        img (image path or np.ndarray): image that lines are drawn on
        lines (list): list of points [[x1, y2, x2, y2], ...] of the lines being drawn
        color (tuple): (x, x, x)
    returns:
        (np.ndarray): image with lines drawn on
    """
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, 3) 
    return img
