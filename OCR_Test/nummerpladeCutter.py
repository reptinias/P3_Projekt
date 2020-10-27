import cv2
import numpy as np
from collections import deque
import pytesseract as pt

letterNumber = 0

def count(img):
    visited = np.zeros((img.shape[0], img.shape[1]))
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if img[y][x] == 0 and visited[y][x] != 1:
                append = True
                XYArray = []
                queue = deque([])
                #nummerpladeFrame[y, x] = (0, 255, 0)
                grassFrie(y, x, XYArray, append, img, visited, queue)
            else:
                visited[y][x] = 1


# GrassFire algorithmen til at finde sorte pixels,
# som tilhøre en større gruppe af sorte pixels
def grassFrie(y, x, XYArray, append, img, visited, queue):

    if append == True:
        visited[y][x] = 1
        XYArray.append([x, y])
        queue.append([x, y])
    append = True

    if x + 1 < img.shape[1] and img[y][x + 1] == 0 and visited[y][x + 1] != 1:
        grassFrie(y, x + 1, XYArray, append, img, visited, queue)

    elif y + 1 < img.shape[0] and img[y + 1][x] == 0 and visited[y + 1][x] != 1:
        grassFrie(y + 1, x, XYArray, append, img, visited, queue)

    elif x - 1 > 0 and img[y][x - 1] == 0 and visited[y][x - 1] != 1:
        grassFrie(y, x - 1, XYArray, append, img, visited, queue)

    elif y - 1 > 0 and img[y - 1][x] == 0 and visited[y - 1][x] != 1:
        grassFrie(y - 1, x, XYArray, append, img, visited, queue)


    elif len(queue) != 0:
        append = False
        x, y = queue.pop()
        grassFrie(y, x, XYArray, append, img, visited, queue)

    else:
        print('jeg er færdig nu')
        xArray, yArray = zip(*XYArray)

        maxX = max(xArray)
        maxY = max(yArray)

        minX = min(xArray)
        minY = min(yArray)

        print(minX)
        print(maxX)
        print(minY)
        print(maxY)

        if minY != maxY and minX != maxX:
            cv2.rectangle(img, (minX, minY), (maxX, maxY), (0, 255, 0), 2)
            letter = img[minY:maxY,  minX:maxX]
            cv2.imshow('letter', letter)



#Load billede og konvater det til gray-scale
#cap = cv2.VideoCapture(0)

#while True:
    # Capture frame-by-frame
    #ret, frame = cap.read()

#indlæs billede
frame = cv2.imread('testnummerplader.jpg')

#grayScaler billede
grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#tiltøj et blur filter
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
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#Laver et frame kun til nummerpladen
#og fylder det med et croped billede
nummerpladeFrame = frame[y: y + h, x: x + w]

threshCrop = np.zeros((nummerpladeFrame.shape[0], nummerpladeFrame.shape[1]), dtype=np.uint8)

for x in range(0, nummerpladeFrame.shape[0]):
    for y in range(0, nummerpladeFrame.shape[1]):
        r, g, b = nummerpladeFrame[x, y]
        gray = (r / 3 + g / 3 + b / 3)

        if gray > 70:
            threshCrop[x, y] = 255
        else:
            threshCrop[x, y] = 0

height = 50
width = int(height * 4.75)
dim = (width, height)
# resize image
resized = cv2.resize(threshCrop, dim, interpolation = cv2.INTER_AREA)
count(resized)

# finder tegn og samler det i en string og printer
out_below = pt.image_to_string(threshCrop)
print(out_below)

cv2.imshow('original', frame)
cv2.imshow('canny', edged)
cv2.imshow('nummplade frame', nummerpladeFrame)
cv2.imshow('thresh nummerplade', threshCrop)
cv2.imshow('resiezed', resized)
cv2.waitKey(0)



#if cv2.waitKey(1) & 0xFF == ord('q'):
    #break
