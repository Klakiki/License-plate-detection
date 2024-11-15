import cv2
import numpy as np
import matplotlib.pyplot as plt

# Convert image
def convertImage(image):
    # convert color-image into gray
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # blur image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # set threshold (turn into black&white img)
    canny = cv2.Canny(blur,100,200)
    return canny

img = cv2.imread("test.jpg")
processed_img = convertImage(img)
origianl_img = img.copy()

contour_img = processed_img.copy()

contours, heirarchy = cv2.findContours(contour_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

for contour in contours:
    p = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.02*p,True)

    if len(approx) == 4:
        x,y,w,h = cv2.boundingRect(contour)
        license_img = origianl_img[y:y+h,x:x+w]
        cv2.imshow("license Detected: ",license_img)
        cv2.drawContours(img, [contour],-1,(0,255,255),2)

cv2.imshow("image",img)
cv2.waitKey(0)