import cv2
import argparse
import numpy as np
import img2pdf
import imutils
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

def checkContourInside(contourList,contour):
    x, y, w, h = cv2.boundingRect(contour)
    for c in contourList:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cx > x and cy > y and cw > w and ch > h:
            return True
    return False

def checkContourOverlaps(contourList,moment):
    cX = int(moment["m10"] / moment["m00"])
    cY = int(moment["m01"] / moment["m00"])
    for c in contourList:
        m = cv2.moments(c)
        mcX = int(m["m10"] / m["m00"])
        mcY = int(m["m01"] / m["m00"])
        if mcX+25 > cX and cX > mcX-25 and mcY+25 > cY and cY > mcY-25:
            print(str(cX)+','+str(cY)+','+str(mcX)+','+str(mcY),cv2.contourArea(c))
            return True, cv2.contourArea(c)
    print(str(cX)+','+str(cY)+','+str(mcX)+','+str(mcY),cv2.contourArea(c))
    return False, cv2.contourArea(c)




def inflateContours(contours,image):
    height, width, channels = image.shape
    print(height,width)
    blank_image = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)
    for c in contours:
        for i in range(len(c)):
            x, y = c[i][0]
            for j in range(5):
                if x+j < width and image[x+j][y][0] == 0 and image[x+j][y][1] == 0:
                    # print(x+j)
                    c[i][0] = x+j, y
                elif y+j < height and image[x][y+j][0] == 0 and image[x][y+j][1] == 0:
                    c[i][0] = x, y+j
                # elif x+j < width and y+j < height and image[x+j][y+j][0] == 0 and image[x+j][y+j][1] == 0:
                    # c[i][0] = x+j, y+j
                elif x-j > 0 and image[x-j][y][0] == 0 and image[x-j][y][1] == 0:
                    # print(x+j)
                    c[i][0] = x-j, y
                elif y-j > 0 and image[x][y-j][0] == 0 and image[x][y-j][1] == 0:
                    c[i][0] = x, y-j
                # elif x-j > 0 and y-j < height and image[x-j][y-j][0] == 0 and image[x-j][y-j][1] == 0:
                    # c[i][0] = x-j, y-j
                
                # blank_image = cv2.drawContours(blank_image, [c], 0, (0, 0, 0), 1)
                # cv2.imshow('image', blank_image)
                # cv2.waitKey(1)
    return contours
            # print(x,y,image[x][y])

def deleteContours(contours):
    pass


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
    # image2 = cv2.dilate(image2,kernel,iterations = 1)
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
    font = cv2.FONT_HERSHEY_SIMPLEX
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
    # edged = cv2.Canny(image2, 30, 50)
    # edged = auto_canny(image2)

    """ c = colors[1]
    region = []
    
    mask = cv2.inRange(image2, c, c)
    res = cv2.bitwise_and(image2, image2, mask=mask)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2:]
    contourImg = image2.copy()
    img = cv2.drawContours(contourImg, contours, -1, (0, 0, 1), 1)
    cv2.imshow('img', image2)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('contours', img)
    cv2.waitKey(0) """


    
    kernel = np.ones((2,2),np.uint8)
    allContours = []
    allM = []
    # allMask = []
    mask = cv2.inRange(image2, colors[0], colors[0])
    for c in colors:
        mask = cv2.inRange(image2, c, c)
        # mask = cv2.GaussianBlur(mask,(1,1),0)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(mask,kernel,iterations = 1)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #res = cv2.bitwise_and(image2, image2, mask=mask)
        # cv2.imshow("Edges", imutils.resize(mask, height=600))
        # cv2.waitKey(0)
        # contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2:]
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contourImg = image2.copy()
        for cnts in contours:
            if cv2.contourArea(cnts) > 500:
                M = cv2.moments(cnts)
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                # img = cv2.drawContours(contourImg, [cnts], 0, (0, 0, 0), 1)
                # fontSize = 0.3
                # allM.append(M)
                allContours.append(cnts)
                # print(cnts)
                # cv2.putText(img, str(np.where(colors == image2[cY, cX])[0][0]+1), (cX, cY),font,fontSize, (255, 255, 255), 1, cv2.LINE_AA)

    
    # allContours = sorted(allContours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    contourImg = image2.copy()
    blank_image = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)
    blank_image = cv2.drawContours(blank_image, allContours, -1, (0, 0, 0), 1)
    # blank_image = cv2.dilate(blank_image,kernel,iterations = 1)
    # allContours,hierarchy = cv2.findContours(blank_image,cv2.CV_RETR_FLOODFILL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    

    for cnts in allContours:
        M = cv2.moments(cnts)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # check,area = checkContourOverlaps(allContours,M)
        img = cv2.drawContours(contourImg, [cnts], 0, (0, 0, 0), 1)
        blank_image = cv2.drawContours(blank_image, [cnts], 0, (0, 0, 0), 1)
        fontSize = 0.3
        # print(cX,cY,cv2.contourArea(cnts))
        # cv2.putText(img, str(np.where(colors == image2[cY, cX])[0][0]+1), (cX, cY),font,fontSize, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(blank_image, str(np.where(colors == image2[cY, cX])[0][0]+1), (cX, cY),font,fontSize, (0, 0, 0), 1, cv2.LINE_AA)
    
    kernel = np.ones((2,2),np.uint8)
    # allContours = inflateContours(allContours,blank_image)
    blank_image2 = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)
    blank_image2 = cv2.drawContours(blank_image, allContours, -1, (0, 0, 0), 1)
    # blank_image2 = cv2.dilate(blank_image2,kernel,iterations = 1)
    # blank_image2 = cv2.morphologyEx(blank_image2, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('inflated_outline.jpg', blank_image2)

    # blank_image = cv2.dilate(blank_image,kernel,iterations = 10)
    # blank_image = cv2.erode(blank_image,kernel,iterations = 10)
    # contours,hierarchy = cv2.findContours(blank_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # blank_image = cv2.drawContours(blank_image, allContours, -1, (0, 0, 0), 1)
    # img = cv2.erode(img,kernel,iterations = 10)
    # blank_image = cv2.dilate(blank_image,kernel,iterations = 10)
    
    
    # blank_image = cv2.morphologyEx(blank_image, cv2.MORPH_CLOSE, kernel)
    # blank_image = cv2.morphologyEx(blank_image, cv2.MORPH_OPEN, kernel)
    # cv2.imshow(img)
    # cv2.waitKey(0)
    # blank_image = cv2.erode(blank_image,kernel,iterations = 10)
    # cv2.imwrite(colored_output, img)
    outline_image_with_no = output_path + "Outline_col_"+str(numColor)+"_det_"+str(detailLevel)+".jpg"
    outline_image = output_path + "Outline_col_"+str(numColor)+"_det_"+str(detailLevel)+"_unnumbered.jpg"
    cv2.imwrite(colored_output, img)
    cv2.imwrite(outline_image, blank_image)
    cv2.imwrite(outline_image_with_no, blank_image)

    pilImg = Image.open(outline_image)
    pilImg2 = Image.open(colored_output)
    pilDraw = ImageDraw.Draw(pilImg)
    pilDraw2 = ImageDraw.Draw(pilImg2)

    for cnts in allContours:
        M = cv2.moments(cnts)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        pilDraw.text((cX, cY), str(np.where(colors == image2[cY, cX])[0][0]+1), font=pilFont,fill=(0,0,0,255))
        pilDraw2.text((cX, cY), str(cX)+','+str(cY)+','+str(cv2.contourArea(cnts)), font=pilFont,fill=(0,0,0,255))

    pilImg.save(outline_image_with_no, "JPEG")
    pilImg2.save(colored_output, "JPEG")

    """ f = open(outline_image, 'w+')
    f.write('<svg width="'+str(width)+'" height="'+str(height)+'" xmlns="http://www.w3.org/2000/svg">')
    for i in range(len(allContours)):
        #print(c[i][0])
        # x, y = contours[i][0][0]
        if cv2.contourArea(allContours[i]) > 0:
            M = cv2.moments(allContours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            f.write('<path d="M ')
            for j in range(len(allContours[i])):
                # x = contours[i][j][0]
                x, y = allContours[i][j][0]
                # print(x, y, len(contours[i][j]))
                f.write(str(x) + ', ' + str(y)+' ')
            f.write('" style="fill:nofill;stroke:gray;stroke-width:1"/>')
            f.write('<text x="'+str(cX)+'" y="'+ str(cY)+'">'+str(np.where(colors == image2[cY, cX])[0][0]+1)+'</text>')

    # f.write('"/>')
    f.write('</svg>')
    f.close() """

    


    # print('regions: ',region)
    # make them thicker
    # kernel = np.ones((2, 2), np.uint8)
    # edged = cv2.morphologyEx(edged, cv2.MORPH_DILATE, kernel)
    # contours,hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2:]
    # contours,hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2:]
    
    # cv2.imshow("Edges", imutils.resize(edged, height=600))


    # cv2.waitKey(0)

    """ print("Total Conturs Found: ",len(contours))
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
    pilImg.save(outline_image_with_no, "JPEG") """
    # pilImg2.save(colored_output, "JPEG")
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

