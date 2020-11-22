from collections import deque

TS = False

if TS:
    import pytesseract as pt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import operator



MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
strFinalString = ""  # declare final string, this will have the final number sequence by the end of the program
# Load original image
imgOriginal = cv2.imread('13020185.jpg')
img2 = cv2.imread('13020185.jpg',0)

class Tesseract:
    def __init__(self, img):
        self.img = img

    def getText(self):
        pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        out_below = pt.image_to_string(self.img, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10')
        print(out_below)



#Empty arrays for the grayscale and binary version of the image
grayImg = np.zeros((imgOriginal.shape[0],imgOriginal.shape[1]))
binaryImg = np.zeros((imgOriginal.shape[0],imgOriginal.shape[1]))

#Convolution function
def convolute(img, filter):
    imgRow, imgCol = img.shape
    kernelRow, kernelCol = filter.shape
    output = np.zeros(img.shape)

    #Add padding to the image
    pad_height = kernelRow // 2
    pad_width = kernelCol // 2
    padded_image = np.zeros((imgRow + (2 * pad_height), imgCol + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img

    for row in range(imgRow):
        for col in range(imgCol):
            output[row, col] = np.sum(filter * padded_image[row:row + kernelRow, col:col + kernelCol])
            output[row, col] /= filter.shape[0] * filter.shape[1]
    return output

#Gray scaling (Step 1)
def grayScale(img):
    print('Running gray scaling')
    height, width, channel = img.shape

    for i in range(height):
        for j in range(width):
            #Convert to grayscale
            r, g, b = img[i, j]
            px = r*0.3 + 0.6*g + 0.1*b
            grayImg[i, j] = px

            #Convert to binary
            if px >= 70:
                binaryImg[i, j] = 255
            else:
                binaryImg[i, j] = 0

    cv2.imwrite('grayImg.jpg', grayImg)
    cv2.imwrite('binaryImg.jpg',binaryImg)

    print('Completed')
    generate_gauss_kernel(5, grayImg)

#Generate 5X5 gaussian kernel (Step 2, pt.1)
def generate_gauss_kernel(size, img, sigma=1):
    print('Generating gaussian kernel')
    kernel = np.zeros((size,size))
    img = img
    for x in range(0,size):
        for y in range(0,size):
            kernel[x,y] = 1/(np.sqrt(2 * np.pi * sigma ** 2)) * np.e ** (-x**2 + y**2/2*sigma**2)
    print('Completed')
    blur(kernel, img)

#Apply the gaussian filter to the image (Step 2, pt.2)
def blur(kernel,img):
    print('Applying filter')
    filteredImage = convolute(img, kernel)

    print('Completed')
    #plt.imshow(filteredImage, cmap='gray')
    #plt.title("Output Image using 5X5 Kernel")
    #plt.show()

    sobel(filteredImage)

#Apply sobel kernels to the filtered image (Step 3)
def sobel(img):
    print('Applying sobel kernel')
    imgRow, imgCol = img.shape
    angle = np.zeros(img.shape)

    Gx = np.array \
        ([[-1,0,1],
          [-2,0,2],
          [-1,0,1]])

    Gy = np.array \
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
    print('Completed')
    non_max(G,angle)

#Non-maximum suppression (Step 4)
def non_max(img, angle):
    print('Applying non-maximum suppression')
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

    print('Completed')
    doubleThreshold(output)

    #plt.imshow(output, cmap='gray')
    #plt.title("Non-maximum suppression")
    #plt.show()

#Double thresholding (Step 5)
def doubleThreshold(img):
    print('Applying double threshold')
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

    print('Completed')
    trackEdge(output)

#Edge tracking (Step 6)
def trackEdge(img):
    print('Tracking edges')
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

    print('Completed')

    cv2.imwrite('edge.jpg',img)
    #plt.imshow(img, cmap='gray')
    #plt.title("Edge tracking")
    #plt.show()

    detectPlate(img)

#License plate detection (Step 7)
def detectPlate(img):
    print('Detecting license plate')
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
    #Resize image
    resized = cv2.resize(nummerpladeFrame, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite('plate.jpg', nummerpladeFrame)
    pl = cv2.imread('plate.jpg')
    print('Completed')

    if TS:
        ts = Tesseract(pl)
        ts.getText()

    plt.imshow(nummerpladeFrame, cmap='gray')
    plt.title("Plate")
    plt.show()

    if not TS:
        print('Detecting characters')
        count(resized)

#Character segmentation (Step 8)
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

    print('Completed')
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
            letter = img[minY:maxY,  minX:maxX]
            letterArray.append(letter)
            plt.imshow(letter, cmap='gray')
            plt.title("Letter")
            plt.show()
            cv2.imwrite('letter.png',letter)





# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def knnresult2():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread("letter.png")          # read in testing numbers image

    if imgTestingNumbers is None:                           # if image was not read successfully
        print ("error: image not read from file \n\n")        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print ("\n" + strFinalString + "\n")                  # show the full string

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    cv2.waitKey(0)                                          # wait for user key press

    cv2.destroyAllWindows()             # remove windows from memory

    return


#grayScale(imgOriginal)
knnresult2()
#sobel(img2)
#nMax = cv2.imread('non_max.jpg',0)
#trackEdge(nMax)

#cv2.imshow('gray',grayimg)
#cv2.waitKey(0)