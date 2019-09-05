import cv2
import argparse
import numpy as np
import imutils


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", imutils.resize(image, height=600))
cv2.waitKey(0)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", imutils.resize(gray, height=600) )
# cv2.waitKey(0)

blurRadii = [(5,5),(7,7),(9,9),(11,11)]
# blurred = cv2.GaussianBlur(gray,blurRadii[0],0)
# cv2.imshow("Blurred", imutils.resize(blurred, height=600) )
# cv2.waitKey(0)

# # Taking a matrix of size 5 as the kernel 
# kernel = np.ones(blurRadii[3], np.uint8)

# img_dilation = cv2.erode(image, kernel, iterations=1) 
# cv2.imshow("Dilation", imutils.resize(blurred, height=600))
# cv2.waitKey(0)



NCLUSTERS = 8
NROUNDS = 4
height, width, channels = image.shape
samples = np.zeros([height*width, 3], dtype=np.float32)
count = 0

for x in range(height):
    for y in range(width):
        samples[count] = image[x][y]  # BGR color
        count += 1

# compactness, labels, centers = cv2.kmeans(samples,NCLUSTERS,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),NROUNDS,cv2.KMEANS_RANDOM_CENTERS)
compactness, labels, centers = cv2.kmeans(samples,NCLUSTERS,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1),NROUNDS,cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
res = centers[labels.flatten()]
image2 = res.reshape((image.shape))


# image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)
# cv2.imshow("KMEANS", image2)
cv2.imshow("KMEANS", imutils.resize(image2, height=600))
cv2.imwrite("kmeans15.jpg", image2)
cv2.waitKey(0)

numLabels = np.arange(0, NCLUSTERS + 1)
(hist, _) = np.histogram(labels, bins=numLabels)
# normalize the histogram, such that it sums to one
hist = hist.astype("float")
hist /= hist.sum()

print(hist)

#appending frequencies to cluster centers
colors = centers

#descending order sorting as per frequency count
colors = colors[(-hist).argsort()]
hist = hist[(-hist).argsort()]

#creating empty chart
# chart = np.zeros((50, 500, 3), np.uint8)
chart = np.zeros((50, 500, 3), np.uint8)
start = 0
print(colors)
#creating color rectangles
for i in range(NCLUSTERS):
    # end = start + int((i+1) * (500/NCLUSTERS))
    end = start + int((i+1) + 50)

    #getting rgb values
    r = colors[i][0]
    g = colors[i][1]
    b = colors[i][2]    

    print(start, end, (r, g, b))
    #using cv2.rectangle to plot colors
    cv2.rectangle(chart, (int(start), 0),(int(end), 50), (int(r), int(g), int(b)), -1)
    start = end


cv2.imshow("Color Pallet", imutils.resize(chart, height=50))
cv2.waitKey(0)

# edged = cv2.Canny(image2, 30, 100)
blurred = cv2.GaussianBlur(image2, (5,5),0)
# edged = auto_canny(blurred)
edged = cv2.Canny(blurred, 30, 50)


cv2.imshow("Edges", imutils.resize(edged, height=600))
cv2.waitKey(0)

contours,hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
print(len(contours))
edged = cv2.bitwise_not(edged)
contourImg = edged.copy()
img = cv2.drawContours(contourImg, contours, -1, (0, 255, 0), 1)
cv2.imshow("Contours", imutils.resize(img, height=600))
cv2.waitKey(0)



cv2.destroyAllWindows()
