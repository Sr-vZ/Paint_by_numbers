import cv2
import argparse
import numpy as np


img = cv2.imread('./static/processed_image/Output_col_8_det_7.jpg')
height, width, channels = img.shape
edged = cv2.Canny(img, 30, 50)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]

# for c in contours:
#     print(c)

print(contours[1][0])

f = open('path.svg', 'w+')
f.write('<svg width="'+str(width)+'" height="'+str(height)+'" xmlns="http://www.w3.org/2000/svg">')
# f.write('<path d="M')
# f.write('<polyline points="')

for i in range(len(contours)):
    #print(c[i][0])
    # x, y = contours[i][0][0]
    if cv2.contourArea(contours[i])>0:
        M = cv2.moments(contours[i])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        f.write('<path d="M ')
        for j in range(len(contours[i])):
            # x = contours[i][j][0]
            x,y = contours[i][j][0]
            # print(x, y, len(contours[i][j]))
            f.write(str(x) + ', ' + str(y)+' ')
        f.write('" style="fill:nofill;stroke:gray;stroke-width:1"/>')
        f.write('<text x="'+str(cX)+'" y="'+ str(cY)+'">'+str(i+1)+'</text>')

# f.write('"/>')
f.write('</svg>')
f.close()
