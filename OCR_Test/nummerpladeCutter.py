import cv2
import numpy as np

#Load billede og konvater det til gray-scale
img = cv2.imread('bilnummerplade2.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Tilføj et blur filter
grayImg = cv2.bilateralFilter(grayImg, 13, 15, 15)

#edge detection
edged = cv2.Canny(grayImg, 50, 200)

#tegner edges på det originale billede
conImg, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
#conImg = sorted(conImg, key = cv2.contourArea(conImg), reverse = True)[:10]

#Viser de forskellige billeder udervejes
cv2.imshow('original', img)
cv2.imshow('gray', grayImg)
cv2.imshow('edged', edged)

cv2.waitKey(0)