import cv2
import argparse
import numpy as np
import img2pdf
import imutils
from PIL import Image, ImageDraw, ImageFont


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# ap.add_argument("-c", "--colors", required=True, help="Number of colors")
# ap.add_argument("-d", "--detail", required=True, help="Level of detail (0-6)")
# args = vars(ap.parse_args())

# Draw voronoi diagram
def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        # print(int(centers[i][0]))
        cX = int(centers[i][0])
        cY = int(centers[i][1])
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = (int(image2[cX, cY][0]), int(image2[cX, cY][1]), int(image2[cX, cY][2]))
        # print(color)
        # cv2.fillConvexPoly(img, ifacet, (255,255,255), cv2.CV_AA, 0)
        # color = (255,255,255)
        cv2.fillConvexPoly(img, ifacet, color)
        ifacets = np.array([ifacet])
        # cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1)
        # cv2.circle(img, (centers[i][0], centers[i][1]),3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)
        # print(ifacets[0])


img = cv2.imread("leafs.JPG")
size = img.shape
rect = (0, 0, size[1], size[0])
subdiv = cv2.Subdiv2D(rect)
# print(subdiv)
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

# all_coords = [(x, y) for x in range(width) for y in range(height)]
# all_colours = [image[x, y] for x, y in all_coords]

# processImage(args["image"], args["colors"], args["detail"])
allContours = []
points = []
for c in colors:
    mask = cv2.inRange(image2, c, c)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnts in contours:
        if cv2.contourArea(cnts) > 100:
            allContours.append(cnts)

# allContours = sorted(allContours, key=lambda x: cv2.contourArea(x), reverse=False)
for c in allContours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    points.append((int(cX), int(cY)))
    # x,y,w,h = cv2.boundingRect(c)
    # points.append((int(x), int(x)))
    # points.append((int(x+w), int(y+h)))
    for i in range(len(c)):
        # print(c[i][0])
        cX, cY = c[i][0]
        points.append((int(cX), int(cY)))

# Insert points into subdiv
# points = [(0,0),(width-1,height-1)]
for p in points:
    subdiv.insert(p)


# Allocate space for Voronoi Diagram
# img_voronoi = np.zeros(img.shape, dtype=img.dtype)
img_voronoi = image2.copy()
# Draw Voronoi diagram
draw_voronoi(img_voronoi, subdiv)

cv2.imshow('vornoi', imutils.resize(img_voronoi, height=600))
cv2.waitKey(0)
