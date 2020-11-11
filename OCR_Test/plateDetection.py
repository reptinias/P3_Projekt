from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

imgOriginal = cv2.imread('bilnummerplade10.jpg')
img2 = cv2.imread('bilnummerplade10.jpg',0)
grayImg = np.zeros((imgOriginal.shape[0],imgOriginal.shape[1]))
binaryImg = np.zeros((imgOriginal.shape[0],imgOriginal.shape[1]))

def convolute(img, filter):
    imgRow, imgCol = img.shape
    kernelRow, kernelCol = filter.shape
    output = np.zeros(img.shape)

    pad_height = kernelRow // 2
    pad_width = kernelCol // 2
    padded_image = np.zeros((imgRow + (2 * pad_height), imgCol + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img

    for row in range(imgRow):
        for col in range(imgCol):
            output[row, col] = np.sum(filter * padded_image[row:row + kernelRow, col:col + kernelCol])
            output[row, col] /= filter.shape[0] * filter.shape[1]

    return output

def grayScale(img):
    height, width, channel = img.shape

    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j]
            px = r*0.3 + 0.6*g + 0.1*b
            grayImg[i, j] = px

            if px >= 70:
                binaryImg[i, j] = 255
            else:
                binaryImg[i, j] = 0

    cv2.imwrite('grayImg.jpg', grayImg)
    cv2.imwrite('binaryImg.jpg',binaryImg)

    print('Gray scaling completed')
    generate_gauss_kernel(5, grayImg)

def generate_gauss_kernel(size, img, sigma=1):
    kernel = np.zeros((size,size))
    img = img
    for x in range(0,size):
        for y in range(0,size):
            kernel[x,y] = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.e ** (-x**2 + y**2/2*sigma**2)
    print('Gaussian kernel generated')
    blur(kernel, img)

def blur(kernel,img):
    filteredImage = convolute(img, kernel)

    print('Blur applied')
    #plt.imshow(filteredImage, cmap='gray')
    #plt.title("Output Image using 5X5 Kernel")
    #plt.show()

    sobel(filteredImage)

def sobel(img):
    imgRow, imgCol = img.shape
    angle = np.zeros(img.shape)

    Gx = np.array\
        ([[-1,0,1],
          [-2,0,2],
          [-1,0,1]])

    Gy = np.array\
        ([[1, 2, 1],
          [0, 0, 0],
          [-1,-2,-1]])

    GxOut = convolute(img, Gx)
    GyOut = convolute(img, Gy)

    for row in range(imgRow):
        for col in range(imgCol):
            angle[row, col] = np.arctan2(GyOut[row, col], GxOut[row, col])

    G = np.sqrt(GxOut**2 + GyOut**2)

    #plt.imshow(G, cmap='gray')
    #plt.title("Sobel")
    #plt.show()
    print("Sobel kernel applied")

    non_max(G,angle)

def non_max(img, angle):
    imgRow, imgCol = img.shape
    output = np.zeros(img.shape)

    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for row in range(1,imgRow-1):
        for col in range(1,imgCol-1):
                q = 255
                r = 255

                if(0 <= angle[row,col] < 22.5) or (157.5 <= angle[row,col] <= 180):
                    q = img[row,col+1]
                    r = img[row,col-1]
                elif (22.5 <= angle[row,col] < 67.5):
                    q = img[row+1,col-1]
                    r = img[row-1,col+1]
                elif (67.5 <= angle[row,col] < 112.5):
                    q = img[row+1,col]
                    r = img[row-1,col]
                elif (112.5 <= angle[row,col] < 157.5):
                    q = img[row-1,col-1]
                    r = img[row+1,col+1]

                if(img[row,col] >= q) and (img[row,col] >= r):
                    output[row,col] = img[row,col]
                else:
                    output[row, col] = 0

    doubleThreshold(output)

    print("Non-maximum suppression")
    #plt.imshow(output, cmap='gray')
    #plt.title("Non-maximum suppression")
    #plt.show()

def doubleThreshold(img):
    row, col = img.shape
    weak = 25
    strong = 255
    output = np.zeros((row,col))

    lowThresholdRatio = 0.05
    highThresholdRatio = 0.2

    highThresh = img.max() * highThresholdRatio
    lowThresh = highThresh * lowThresholdRatio

    weakX, weakY = np.where((img <= highThresh) & (img >= lowThresh))
    strongX, strongY = np.where(img >= highThresh)

    output[strongX, strongY] = strong
    output[weakX, weakY] = weak
    #cv2.imwrite('non_max.jpg',output)
    trackEdge(output)

def trackEdge(img):
    strong = 255
    weak = 25

    for row in range(1, img.shape[0]-1):
        for col in range(1, img.shape[1]-1):
            if (img[row,col] == weak):
                try:
                    if (img[row+1, col] == strong or img[row, col+1] == strong or img[row-1, col] == strong
                            or img[row, col-1] == strong or img[row+1, col+1] == strong
                            or img[row-1, col-1] == strong or img[row+1, col-1] == strong
                            or img[row-1, col+1] == strong):
                        img[row, col] = 255
                    else:
                        img[row, col] = 0
                except IndexError as e:
                    print("ERR")

    cv2.imwrite('edge.jpg',img)
    plt.imshow(img, cmap='gray')
    plt.title("Edge tracking")
    plt.show()

    detectPlate(img)

def detectPlate(img):
    img = img.astype(np.uint8)
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(binaryImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

    nummerpladeFrame = binaryImg[y: y + h, x: x + w]

    height = 50
    width = int(height * 4.75)
    dim = (width, height)
    # resize image
    resized = cv2.resize(nummerpladeFrame, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite('plate.jpg', nummerpladeFrame)

    count(resized)

    plt.imshow(nummerpladeFrame, cmap='gray')
    plt.title("Plate")
    plt.show()

def count(img):
    visited = np.zeros((img.shape[0], img.shape[1]))
    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if img[y][x] == 0 and visited[y][x] != 1:
                append = True
                XYArray = []
                queue = deque([])
                grassFire(y, x, XYArray, append, img, visited, queue)
            else:
                visited[y][x] = 1

# GrassFire algorithmen til at finde sorte pixels,
# som tilhøre en større gruppe af sorte pixels
letterArray = []

def grassFire(y, x, XYArray, append, img, visited, queue):

    if append == True:
        visited[y][x] = 1
        XYArray.append([x, y])
        queue.append([x, y])
    append = True

    if x + 1 < img.shape[1] and img[y][x + 1] == 0 and visited[y][x + 1] != 1:
        grassFire(y, x + 1, XYArray, append, img, visited, queue)

    elif y + 1 < img.shape[0] and img[y + 1][x] == 0 and visited[y + 1][x] != 1:
        grassFire(y + 1, x, XYArray, append, img, visited, queue)

    elif x > 0 and img[y][x - 1] == 0 and visited[y][x - 1] != 1:
        grassFire(y, x - 1, XYArray, append, img, visited, queue)

    elif y > 0 and img[y - 1][x] == 0 and visited[y - 1][x] != 1:
        grassFire(y - 1, x, XYArray, append, img, visited, queue)

    elif len(queue) != 0:
        append = False
        x, y = queue.pop()
        grassFire(y, x, XYArray, append, img, visited, queue)

    else:
        xArray, yArray = zip(*XYArray)

        maxX = max(xArray)
        maxY = max(yArray)

        minX = min(xArray)
        minY = min(yArray)


        if maxY - minY > 15 and maxX - minX > 10:
            cv2.rectangle(img, (minX, minY), (maxX, maxY), (0, 255, 0), 1)
            letter = img[minY:maxY,  minX:maxX]
            letterArray.append(letter)

            plt.imshow(letter, cmap='gray')
            plt.title("Letter")
            plt.show()


grayScale(imgOriginal)
#sobel(img2)
#nMax = cv2.imread('non_max.jpg',0)
#trackEdge(nMax)

#cv2.imshow('gray',grayimg)
#cv2.waitKey(0)