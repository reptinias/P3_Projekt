import cv2
import numpy as np
import matplotlib.pyplot as plt

img2 = cv2.imread('bilnummerplade10.jpg',0)
img1 = cv2.imread('imgT.jpg',0)
grayimg = img1


def grayScale(image):
    height, width = img2.shape
    for i in range(height):
        for j in range(width):
            r,g,b = image[i,j]
            px = r*0.3 + 0.6*g + 0.1*b
            grayimg[i,j] = px

def generate_gauss_kernel(size, sigma=1):
    kernel = np.zeros((size,size))

    for x in range(0,size):
        for y in range(0,size):
            kernel[x,y] = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.e ** (-x**2 + y**2/2*sigma**2)
    print(kernel)
    blur(kernel, img2)

def blur(kernel,image):
    imgRow, imgCol = image.shape
    kernelRow, kernelCol = kernel.shape

    output = np.zeros(image.shape)

    pad_height = kernelRow // 2
    pad_width = kernelCol // 2
    padded_image = np.zeros((imgRow + (2 * pad_height), imgCol + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(imgRow):
        for col in range(imgCol):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernelRow, col:col + kernelCol])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

    plt.imshow(output, cmap='gray')
    plt.title("Output Image using {}X{} Kernel".format(kernelRow, kernelCol))
    plt.show()

#grayScale(img1)
generate_gauss_kernel(5)


#cv2.imshow('gray',grayimg)
cv2.waitKey(0)