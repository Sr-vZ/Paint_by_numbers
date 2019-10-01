import cv2
import argparse
import numpy as np
import imutils
from scipy.interpolate import splprep, splev

img = cv2.imread("leafs.JPG")
NCLUSTERS = 8
NROUNDS = 4
height, width, channels = img.shape
samples = np.zeros([height*width, 3], dtype=np.float32)
count = 0
for x in range(height):
    for y in range(width):
        samples[count] = img[x][y]  # BGR color
        count += 1

compactness, labels, centers = cv2.kmeans(samples,NCLUSTERS,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1),NROUNDS,cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
# print(len(labels))
res = centers[labels.flatten()]
image2 = res.reshape((img.shape))
# image2 = cv2.dilate(image2,kernel,iterations = 1)
# colored_output = 
# cv2.imwrite(colored_output, image2)
numLabels = np.arange(0, NCLUSTERS + 1)
(hist, _) = np.histogram(labels, bins=numLabels)
# normalize the histogram, such that it sums to one
hist = hist.astype("float")
hist /= hist.sum()

#appending frequencies to cluster centers
colors = centers


#descending order sorting as per frequency count
colors = colors[(-hist).argsort()]
hist = hist[(-hist).argsort()]

cv2.imshow('K-means', imutils.resize(image2, height=600))
cv2.waitKey(0)

region = []
color = colors[0]

# # print(color,image2[0][1])
# for x in range(width):
#     for y in range(height):
#         # for c in colors:
#         print(x, y, color)
#         if np.array_equal(color,image2[x][y]):
#             # print(x,y)
#             region.append((x,y))
# print(region)
blank_image = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)
allContours = []
allApprox = []
for c in colors:
    mask = cv2.inRange(image2, c, c)
    # res = cv2.bitwise_or(image2, image2, mask=mask)
    res = cv2.bitwise_not(mask)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]    
    for cnts in contours:
        if cv2.contourArea(cnts) > 300:
            M = cv2.moments(cnts)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            allContours.append(cnts)            
            # blank_image = cv2.drawContours(blank_image, [cnts], 0, (0, 1, 0), 1)
            # blank_image = cv2.morphologyEx(blank_image,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
            # blank_image = cv2.circle(blank_image, (cX, cY), 5, (0, 0, 255),1)
            # cv2.imshow('Contours', imutils.resize(blank_image, height=600))
            # cv2.waitKey(100)
            perimeter = cv2.arcLength(cnts, True)
            epsilon = 0.0005*cv2.arcLength(cnts, True)
            approx = cv2.approxPolyDP(cnts, epsilon, True)
            allApprox.append(approx)


image2 = cv2.drawContours(image2, allApprox, -1, (0, 0, 0), 1)
# image2 = cv2.morphologyEx(image2,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)))
cv2.imshow('Contours', imutils.resize(image2, height=600))
cv2.imwrite('mytest1.jpg', image2)
cv2.waitKey(0)
blank_image = cv2.drawContours(blank_image, allApprox, -1, (0, 0, 0), 1)
# blank_image = cv2.morphologyEx(blank_image,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)))
cv2.imshow('Morphed Contours', imutils.resize(blank_image, height=600))
cv2.imwrite('mytest2.jpg', blank_image)
cv2.waitKey(0)
    # cv2.imshow('Mask', imutils.resize(mask, height=600))
    # cv2.waitKey(0)

