import cv2
import argparse
import numpy as np
import img2pdf
import copy
# import imutils
from PIL import Image, ImageDraw, ImageFont

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged


def neighborsSame(image,x,y):
    height, width, channels = image.shape
    val = image[x][y]
    xRel = [1, 0]
    yRel = [0, 1]
    for i in range(len(xRel)):
        xx = x + xRel[i]
        yy = y + yRel[i]
        if (xx >= 0 and  xx < width and yy >= 0 and yy < height):
            if (image[yy][xx] != val):
                return false
    return true

def outline(image):
    height, width, channels = image.shape
    line = []
    for i in range(width):
        line[i] = [0]*width

    for y in range(height):
        for x in range(width):
            line[y][x] = if neighborsSame(image, x, y) 0 else 1


def getRegion(image, cov, x, y):
    covered = copy.deepcopy(cov)
    region = {value: image[y][x], x: [], y: []}
    value = image[y][x]
    queue = [[x, y]]
    while len(queue) > 0:
        coord = queue.shift()
        if (covered[coord[1]][coord[0]] == false and mat[coord[1]][coord[0]] == value):
            region.x.push(coord[0])
            region.y.push(coord[1])
            covered[coord[1]][coord[0]] = true
            if (coord[0] > 0):
                queue.push([coord[0] - 1, coord[1]])
            if (coord[0] < mat[0].length - 1):
                 queue.push([coord[0] + 1, coord[1]])
            if (coord[1] > 0):
                 queue.push([coord[0], coord[1] - 1])
            if (coord[1] < mat.length - 1):
                queue.push([coord[0], coord[1] + 1])
    return region


def coverRegion(covered, region):
    for i in range(len(region.x)):
        covered[region.y[i]][region.x[i]] = true


def sameCount(image, x, y, incX, incY):
    value = image[y][x]
    count = -1
    while (x >= 0 and x < image[0].length and y >= 0 and y < image.length and image[y][x] == value):
        count = count+1
	    x = x + incX
	    y = y + incY
    return count


def getLabelLoc(image,region):
    bestI = 0
    best = 0
    for i range(len(region.x)):
        goodness = sameCount(image, region.x[i], region.y[i], -1, 0) *  sameCount(image, region.x[i], region.y[i], 1, 0) * sameCount(image, region.x[i], region.y[i], 0, -1) *   sameCount(image, region.x[i], region.y[i], 0, 1)
    if goodness > best:
        best = goodness
        bestI = i
    return {
	value: region.value,
	x: region.x[bestI],
	y: region.y[bestI]
    }




output_path ='./static/processed_image/'

def processImage(srcImage,numColor,detailLevel):
    print("Image : {0} No of Colors: {1} Detail Level: {2}".format(srcImage,numColor,detailLevel))
    image = cv2.imread(srcImage)
    blurRadii = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (13, 13),(15,15)]
    gblurRadii = [(3, 3), (4, 4), (5, 5), (7, 7), (9, 9), (11, 11), (13, 13),(15,15)]
    kernel = np.ones(blurRadii[int(detailLevel)], np.uint8)
    # image = cv2.blur(image, blurRadii[int(detailLevel)])
    image = cv2.GaussianBlur(image, gblurRadii[int(detailLevel)], 0)
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
    colored_output = output_path + "Output_col_"+str(numColor)+"_det_"+str(detailLevel)+".jpg"
    cv2.imwrite(colored_output, image2)
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
    font = cv2.FONT_HERSHEY_DUPLEX
    # ft = cv2.freetype.createFreeType2()
    # ft.loadFontData(fontFileName='Roboto-Black.ttf',id=0)
    pilFont = ImageFont.truetype('RobotoMono-Regular.ttf', size=10)
    for i in range(NCLUSTERS):
        # end = start + int((i+1) * (500/NCLUSTERS))
        end = start + 50

        #getting rgb values
        r = colors[i][0]
        g = colors[i][1]
        b = colors[i][2]

        # print(start, end, (r, g, b))
        #using cv2.rectangle to plot colors
        cv2.rectangle(chart, (int(start), 0), (int(end), 50),(int(r), int(g), int(b)), -1)
        cv2.putText(chart, str(i+1), (int(start)+25, 25),font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        start = end


    # cv2.imshow("Color Pallet", imutils.resize(chart, height=50))
    color_palette = output_path + "Color Palette.jpg"
    cv2.imwrite(color_palette, chart)

    #find the edges in the image
    edged = cv2.Canny(image2, 30, 50)
    # edged = auto_canny(image2)

    # make them thicker
    # kernel = np.ones((2, 2), np.uint8)
    # edged = cv2.morphologyEx(edged, cv2.MORPH_DILATE, kernel)
    # contours,hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2:]
    contours,hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2:]
    # cv2.imshow("Edges", imutils.resize(edged, height=600))


    cv2.waitKey(0)

    print("Total Conturs Found: ",len(contours))
    edged = cv2.bitwise_not(edged)
    contourImg = edged.copy()
    # img = cv2.drawContours(contourImg, contours, -1, (0, 0, 0), 1)
    for c in contours:
        if(cv2.contourArea(c) > 250):
            img = cv2.drawContours(contourImg, [c], 0, (0, 0, 0), 1)
    # cv2.imshow("Contours", imutils.resize(img, height=600))
    # cv2.waitKey(0)
    outline_image = output_path + "Outline_col_"+str(numColor)+"_det_"+str(detailLevel)+"_unnumbered.jpg"
    cv2.imwrite(outline_image, img)
    i = 0
    maxContour = max(contours, key=cv2.contourArea)
    minContour = min(contours, key=cv2.contourArea)
    pilImg = Image.open(outline_image)
    pilImg2 = Image.open(colored_output)
    pilDraw = ImageDraw.Draw(pilImg)
    pilDraw2 = ImageDraw.Draw(pilImg2)
    for c in contours:
        if(cv2.contourArea(c)>250):            
            x, y, w, h = cv2.boundingRect(c)
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            i = i+1
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # fontSize = cv2.contourArea(c)/cv2.contourArea(maxContour)
            fontSize = 0.4
            
            color=(0, 0, 0)
            # pilDraw.text((cX, cY), str(np.where(colors == image2[cY, cX])[0][0]+1), font=pilFont)
            pilDraw.text((cX, cY), str(i), font=pilFont)
            pilDraw2.text((cX, cY), str(i), font=pilFont)
            # cv2.putText(img, str(np.where(colors == image2[y+int(h/2), x+int(w/2)])[
            #             0][0]+1), (x+int(w/2), y+int(h/2)), font, fontSize, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, str(np.where(colors == image2[cY, cX])[0][0]+1), (cX, cY),font,fontSize, (0, 0, 0), 1, cv2.LINE_AA)
            print('Contour: '+str(i)+' Area: '+str(cv2.contourArea(c))+' ratio: '+str(cv2.contourArea(c)/cv2.contourArea(maxContour)))
    
    print('Total Area Outlined: ',i)
            # print(np.where(colors == image2[y+int(h/2), x+int(w/2)]))
    # print(contours[1], cv2.contourArea(contours[1]))
     
    outline_image_with_no = output_path + "Outline_col_"+str(numColor)+"_det_"+str(detailLevel)+".jpg"
    cv2.imwrite(outline_image_with_no, img)
    pilImg.save(outline_image_with_no, "JPEG")
    pilImg2.save(colored_output, "JPEG")
    # cv2.imshow("Contour Text", imutils.resize(img, height=600))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    output_pdf = output_path+' Output.pdf'
    with open(output_pdf, 'wb') as f:
        f.write(img2pdf.convert([srcImage, colored_output, outline_image, outline_image_with_no, color_palette]))
    f.close()

    return colored_output, color_palette, outline_image_with_no, output_pdf

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the input image")
ap.add_argument("-c", "--colors", required=True, help="Number of colors")
ap.add_argument("-d", "--detail", required=True, help="Level of detail (0-6)")
args = vars(ap.parse_args())
processImage(args["image"], args["colors"], args["detail"])

