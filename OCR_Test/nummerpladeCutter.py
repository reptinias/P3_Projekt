import cv2
import numpy as np
import pytesseract as pt

#Load billede og konvater det til gray-scale
img = cv2.imread('bilnummerplade7.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Tilføj et blur filter
grayImg = cv2.bilateralFilter(grayImg, 13, 15, 15)

#edge detection
edged = cv2.Canny(grayImg, 50, 200)

#Finder contours af det originale  billede
_, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Læse mere on contours

#Vi finder indexet for det største contour area
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]

#Finder højden og breden af den rektangel, som contouren tegner
x, y, w, h = cv2.boundingRect(cnt) #Læser mere om dette
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

#Laver et frame kun til nummerpladen
#og fylder det med et croped billede
nummerpladeFrame = img[y: y + h, x: x + w]

#Laver et binart billede af det croped billede
grayCrop = cv2.cvtColor(nummerpladeFrame, cv2.COLOR_BGR2GRAY)
_, threshCrop = cv2.threshold(grayCrop, 150, 255, cv2.THRESH_BINARY_INV)

#finder tegn og samler det i en string og printer
out_below = pt.image_to_string(threshCrop)
print("output: ", out_below)

#Viser de forskellige billeder udervejes
cv2.imshow('original', img)
cv2.imshow('threshCrop', threshCrop)
cv2.imshow('crop', nummerpladeFrame)

cv2.waitKey(0)