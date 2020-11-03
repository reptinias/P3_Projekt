import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bilnummerplade10.jpg')
img2 = cv2.imread('bilnummerplade10.jpg',0)

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
    grayImg = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            r,g,b = img[i,j]
            px = r*0.3 + 0.6*g + 0.1*b
            grayImg[i,j] = px
    print('Grey scaling completed')
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
    plt.imshow(filteredImage, cmap='gray')
    plt.title("Output Image using 5X5 Kernel")
    plt.show()

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

    plt.imshow(G, cmap='gray')
    plt.title("Sobel")
    plt.show()

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

    plt.imshow(output, cmap='gray')
    plt.title("Non-maximum suppression")
    plt.show()


#grayScale(img)
sobel(img2)

#cv2.imshow('gray',grayimg)
#cv2.waitKey(0)