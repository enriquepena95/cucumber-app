import cv2
import numpy as np
import pandas as pd



def findArucoMarkers(img, markerSize = 7, totalMarkers = 50, draw = True):
    global pixelsPerMetric
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.getPredefinedDictionary(key)
    arucoParam = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)
    bboxs, ids, rejected = detector.detectMarkers(gray)
    arucofound =  bboxs, ids, rejected
    aruco_perimeter= cv2.arcLength(arucofound[0][0][0], True)
    pixelsPerMetric = aruco_perimeter / 200
    print("pixel to mm", pixelsPerMetric)
    return pixelsPerMetric