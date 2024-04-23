import cv2
import numpy as np


def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[1] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

    return ver


def rectContour(contours):
    rectCont = []
    for i in contours:
        area = cv2.contourArea(i)
        # print("Area:\n",area)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print("Corner Points:", len(approx))
            if len(approx) == 4:
                rectCont.append(i)
    rectCont = sorted(rectCont, key=cv2.contourArea, reverse=True)

    return rectCont


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx


def reorder(myPoints):
    # We have the Shape of the points as (4,1,2) which means 4 rows and each row has 2 points x,y but the middle 1 is
    # redundant therefore Reshape
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # We add the points x and y part and the smallest one will be the origin
    # print(myPoints)
    # print(add)
    myPointsNew[0] = myPoints[np.argmin(add)]  # [0, 0]
    myPointsNew[3] = myPoints[np.argmax(add)]  # [w, h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # [w, 0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [0, h]
    # print(diff)

    return myPointsNew


def splitBoxes(img, ques, choices):
    rows = np.vsplit(img, ques)
    # cv2.imshow("Split", rows[0])
    boxes = []
    for r in rows:
        cols = np.hsplit(r, choices)

        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split", box)

    return boxes


def showAnswers(img, questions, choices, myIndex, grading, ans):
    # print(img.shape)
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)
    # print(secH,secW)

    for x in range(questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2

        correctColor = (0, 255, 0)
        wrongColor = (0, 0, 255)
        if grading[x] == 1:
            myColor = correctColor
        else:
            myColor = wrongColor
            correctAns = ans[x]
            cX1 = (correctAns * secW) + secW // 2
            # cY1 = (x * secH) + secH // 2
            cv2.circle(img, (cX1, cY), 15, correctColor, cv2.FILLED)

        cv2.circle(img, (cX, cY), 30, myColor, cv2.FILLED)

    return img
