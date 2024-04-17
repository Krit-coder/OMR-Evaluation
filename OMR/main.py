import cv2
import numpy as np
import utils

path = "4.jpg"
widthImg, heightImg = 700, 700

# PREPROCESSING
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# Finding all contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

# FIND RECTANGLES
rectCont = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCont[0])
gradePoints = utils.getCornerPoints(rectCont[2])
# print(biggestContour)

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    p1 = np.float32(biggestContour)
    p2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(p1, p2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))


imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imgBlank])
imgStacked = utils.stackImages(imageArray, 0.5)

cv2.imshow("Stacked Images", imgStacked)
cv2.waitKey(0)
