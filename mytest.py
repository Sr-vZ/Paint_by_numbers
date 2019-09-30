import cv2
import argparse
import numpy as np
import imutils


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

for c in colors:
    mask = cv2.inRange(image2, c, c)
    # res = cv2.bitwise_or(image2, image2, mask=mask)
    res = cv2.bitwise_not(mask)
    cv2.imshow('Mask', imutils.resize(res, height=600))
    cv2.waitKey(0)

