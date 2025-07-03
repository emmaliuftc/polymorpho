import math
import cv2 as cv
import numpy as np

def show(x,y):
    cv.imshow(y,x)
    cv.waitKey(0)
    cv.destroyAllWindows()


print("Hello World")
whiteonblack = cv.imread('probability_map-2.png')
blackonwhite = cv.imread('probability_map-1.png')
whiteonblack = cv.cvtColor(whiteonblack,cv.COLOR_BGR2GRAY)
print(f"data type of image = {whiteonblack.dtype}")
amean = cv.adaptiveThreshold(whiteonblack,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,21,1)
agauss = cv.adaptiveThreshold(whiteonblack,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,21,1)
# show(amean,"adaptive mean thresholding")
# show(agauss,"adaptive gaussian thresholding")
thresh = cv.threshold(whiteonblack,127,255,cv.THRESH_BINARY)[1]
show(thresh,"binary")
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))

thresh = cv.erode(thresh,kernel,iterations=1)
show(thresh,"after eroding")

# morph1 = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel)

# show(morph1,"after closing")
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
morph2 = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel)

show(morph2,"after opening")

# show(whiteonblack)
# kernel = np.ones((5,5),np.uint8)
# closing = cv.morphologyEx(whiteonblack,cv.MORPH_CLOSE,kernel)
# show(closing)
# show(blackonwhite)
# closing = cv.morphologyEx(blackonwhite,cv.MORPH_CLOSE,kernel)
# show(closing)