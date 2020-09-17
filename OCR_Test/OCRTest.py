import cv2
import numpy as np
import pytesseract as pt

img = cv2.imread('testnummerplader.jpg')
cv2.imshow('nummerplade', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

gray, img_bin = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
cv2.imshow('black/white', img_bin)

gray = cv2.bitwise_not(img_bin)
cv2.imshow('invert', gray)

kernel = np.ones((2,1), np.uint8)
img = cv2.erode(gray, kernel, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)
out_below = pt.image_to_string(img)

print("output: ", out_below)
cv2.imshow('res', img)

cv2.waitKey(0)