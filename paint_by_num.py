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
ap.add_argument("-c", "--colors", required=True, help="Number of colors")
ap.add_argument("-d", "--detail", required=True, help="Level of detail (0-6)")
args = vars(ap.parse_args())


def processImage(srcImage,numColor,detailLevel):
    print("Image : {0} No of Colors: {1} Detail Level: {2}".format(srcImage,numColor,detailLevel))
    image = cv2.imread(srcImage)
    blurRadii = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (13, 13)]
    kernel = np.ones(blurRadii[int(detailLevel)], np.uint8)
    image = cv2.blur(image, blurRadii[int(detailLevel)])
    NCLUSTERS = int(numColor)
    NROUNDS = 4
    height, width, channels = image.shape
    samples = np.zeros([height*width, 3], dtype=np.float32)
    count = 0
    for x in range(height):
        for y in range(width):
            samples[count] = image[x][y]  # BGR color
            count += 1
    
    compactness, labels, centers = cv2.kmeans(samples,NCLUSTERS,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1),NROUNDS,cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    image2 = res.reshape((image.shape))
    cv2.imwrite("Output_col_"+numColor+"_det_"+detailLevel+".jpg", image2)
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

    #creating empty chart
    # chart = np.zeros((50, 500, 3), np.uint8)
    chart = np.zeros((50, 50*NCLUSTERS, 3), np.uint8)
    start = 0
    print(colors)
    #creating color rectangles
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(NCLUSTERS):
        # end = start + int((i+1) * (500/NCLUSTERS))
        end = start + 50

        #getting rgb values
        r = colors[i][0]
        g = colors[i][1]
        b = colors[i][2]

        # print(start, end, (r, g, b))
        #using cv2.rectangle to plot colors
        cv2.rectangle(chart, (int(start), 0), (int(end), 50),
                    (int(r), int(g), int(b)), -1)
        cv2.putText(chart, str(i+1), (int(start)+25, 25),
                    font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        start = end


    # cv2.imshow("Color Pallet", imutils.resize(chart, height=50))
    cv2.imwrite("Color Palette.jpg", chart)

    #find the edges in the image
    edged = cv2.Canny(image2, 30, 50)

    contours,hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2:]
    # cv2.imshow("Edges", imutils.resize(edged, height=600))


    cv2.waitKey(0)

    print("Total Conturs Found: ",len(contours))
    edged = cv2.bitwise_not(edged)
    contourImg = edged.copy()
    img = cv2.drawContours(contourImg, contours, -1, (0, 0, 0), 1)
    # cv2.imshow("Contours", imutils.resize(img, height=600))
    # cv2.waitKey(0)

    cv2.imwrite("Outline_col_"+numColor+"_det_"+detailLevel+"_unnumbered.jpg", img)
    i = 0
    for c in contours:
        if(cv2.contourArea(c)>10):

            x, y, w, h = cv2.boundingRect(c)
            i = i+1
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(img, str(i), (x+int(w/2), y+int(h/2)),
                        font, .5, (0, 0, 255), 1, cv2.LINE_AA)
            print(image2[y+int(h/2), x+int(w/2)] in colors)
    # print(contours[1], cv2.contourArea(contours[1]))
    cv2.imwrite("Outline_col_"+numColor+"_det_"+detailLevel+".jpg", img)
    # cv2.imshow("Contour Text", imutils.resize(img, height=600))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


processImage(args["image"], args["colors"], args["detail"])
