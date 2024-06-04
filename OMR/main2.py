import cv2
import numpy as np
import utils

score, rollNo = 0, 0


def check(img):
    ################################
    path = "9.jpg"
    widthImg, heightImg = 700, 700
    questions = 10
    choices = 4
    # ans = [1, 2, 1, 3, 0, 1, 1, 3, 2, 1, 0, 3, 1, 2, 3, 2, 2, 2, 0, 2, 1, 3, 0, 2, 1, 3, 2, 2, 2, 0, 0, 0, 1, 1, 1, 3, 1, 2,
    #        1, 3]
    ans1 = [1, 2, 1, 3, 0, 1, 1, 3, 2, 1]
    ans2 = [0, 3, 1, 2, 3, 2, 2, 2, 0, 2]
    ans3 = [1, 3, 0, 2, 1, 3, 2, 2, 2, 0]
    ans4 = [0, 0, 1, 1, 1, 3, 1, 2, 1, 3]
    ################################

    # PREPROCESSING
    img = cv2.imread(path)
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBubbleContours = img.copy()
    imgFinal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # Finding all contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

    # FIND RECTANGLES
    rectCont = utils.rectContour(contours)
    rollContour = utils.getCornerPoints(rectCont[0])
    gradePoints = utils.getCornerPoints(rectCont[6])
    bubbleContour1 = utils.getCornerPoints(rectCont[3])
    bubbleContour2 = utils.getCornerPoints(rectCont[2])
    bubbleContour3 = utils.getCornerPoints(rectCont[4])
    bubbleContour4 = utils.getCornerPoints(rectCont[1])
    # print(bubbleContour)

    if (bubbleContour1.size != 0 and bubbleContour2.size != 0 and
            rollContour.size != 0 and gradePoints.size != 0):
        cv2.drawContours(imgBubbleContours, bubbleContour1, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBubbleContours, bubbleContour2, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBubbleContours, bubbleContour3, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBubbleContours, bubbleContour4, -1, (0, 255, 0), 20)
        cv2.drawContours(imgBubbleContours, rollContour, -1, (0, 0, 255), 20)
        cv2.drawContours(imgBubbleContours, gradePoints, -1, (255, 0, 0), 20)

        bubbleContour1 = utils.reorder(bubbleContour1)
        bubbleContour2 = utils.reorder(bubbleContour2)
        bubbleContour3 = utils.reorder(bubbleContour3)
        bubbleContour4 = utils.reorder(bubbleContour4)
        rollContour = utils.reorder(rollContour)
        gradePoints = utils.reorder(gradePoints)

        p11 = np.float32(bubbleContour1)
        p12 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(p11, p12)
        imgWarpColored1 = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        p21 = np.float32(bubbleContour2)
        p22 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(p21, p22)
        imgWarpColored2 = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        p31 = np.float32(bubbleContour3)
        p32 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(p31, p32)
        imgWarpColored3 = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        p41 = np.float32(bubbleContour4)
        p42 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(p41, p42)
        imgWarpColored4 = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        pr1 = np.float32(rollContour)
        pr2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrixR = cv2.getPerspectiveTransform(pr1, pr2)
        imgRollWarpColored = cv2.warpPerspective(img, matrixR, (widthImg, heightImg))
        # cv2.imshow("Roll", imgGradeDisplay)

        pg1 = np.float32(gradePoints)
        pg2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(pg1, pg2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
        # cv2.imshow("Grade", imgGradeDisplay)

        # APPLY THRESHOLD
        imgWarpGary1 = cv2.cvtColor(imgWarpColored1, cv2.COLOR_BGR2GRAY)
        imgWarpGary2 = cv2.cvtColor(imgWarpColored2, cv2.COLOR_BGR2GRAY)
        imgWarpGary3 = cv2.cvtColor(imgWarpColored3, cv2.COLOR_BGR2GRAY)
        imgWarpGary4 = cv2.cvtColor(imgWarpColored4, cv2.COLOR_BGR2GRAY)
        imgThresh1 = cv2.threshold(imgWarpGary1, 170, 255, cv2.THRESH_BINARY_INV)[1]
        imgThresh2 = cv2.threshold(imgWarpGary2, 170, 255, cv2.THRESH_BINARY_INV)[1]
        imgThresh3 = cv2.threshold(imgWarpGary3, 170, 255, cv2.THRESH_BINARY_INV)[1]
        imgThresh4 = cv2.threshold(imgWarpGary4, 170, 255, cv2.THRESH_BINARY_INV)[1]
        imgRollWarpGray = cv2.cvtColor(imgRollWarpColored, cv2.COLOR_BGR2GRAY)
        imgRollThresh = cv2.threshold(imgRollWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        boxes1 = utils.splitBoxes(imgThresh1, questions, choices)
        boxes2 = utils.splitBoxes(imgThresh2, questions, choices)
        boxes3 = utils.splitBoxes(imgThresh3, questions, choices)
        boxes4 = utils.splitBoxes(imgThresh4, questions, choices)
        # cv2.imshow("Boxes:", boxes[0])
        # cv2.imshow("Boxes:", boxes[2])
        boxesR = utils.splitBoxes(imgRollThresh, 10, 7)
        # print(boxesR)
        # print(cv2.countNonZero(boxes[0]), cv2.countNonZero(boxes[1]), cv2.countNonZero(boxes[2]), cv2.countNonZero(
        #     boxes[3]), cv2.countNonZero(boxes[4]))

        # GETTING NON-ZERO PIXEL VALUES OF EACH BOX OF MARKINGS
        myPixelVal1 = np.zeros((questions, choices))
        countC = 0
        countR = 0
        for image in boxes1:
            totalPixels1 = cv2.countNonZero(image)
            myPixelVal1[countR][countC] = totalPixels1
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0
        # print(myPixelVal1)

        myPixelVal2 = np.zeros((questions, choices))
        countC = 0
        countR = 0
        for image in boxes2:
            totalPixels2 = cv2.countNonZero(image)
            myPixelVal2[countR][countC] = totalPixels2
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0
        # print(myPixelVal)

        myPixelVal3 = np.zeros((questions, choices))
        countC = 0
        countR = 0
        for image in boxes3:
            totalPixels3 = cv2.countNonZero(image)
            myPixelVal3[countR][countC] = totalPixels3
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0
        # print(myPixelVal)

        myPixelVal4 = np.zeros((questions, choices))
        countC = 0
        countR = 0
        for image in boxes4:
            totalPixels4 = cv2.countNonZero(image)
            myPixelVal4[countR][countC] = totalPixels4
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0
        # print(myPixelVal)

        # GETTING NON-ZERO PIXEL VALUES OF EACH BOX OF ROLL NUMBER
        myPixelValRoll = np.zeros((10, 7))
        rollCountC = 0
        rollCountR = 0
        for image in boxesR:
            totalPixelsRoll = cv2.countNonZero(image)
            myPixelValRoll[rollCountR][rollCountC] = totalPixelsRoll
            rollCountC += 1
            if rollCountC == 7:
                rollCountR += 1
                rollCountC = 0

        # print(myPixelValRoll)
        # print(np.transpose(myPixelValRoll))

        # FINDING INDEX VALUES OF THE MARKINGS
        myIndex1 = []
        for x in range(questions):
            arr = myPixelVal1[x]
            myIndexVal1 = np.where(arr == np.amax(arr))
            # print(myIndexVal[0])
            myIndex1.append(myIndexVal1[0][0])
        # print(myIndex)
        myIndex2 = []
        for x in range(questions):
            arr = myPixelVal2[x]
            myIndexVal2 = np.where(arr == np.amax(arr))
            # print(myIndexVal[0])
            myIndex2.append(myIndexVal2[0][0])
        # print(myIndex)
        myIndex3 = []
        for x in range(questions):
            arr = myPixelVal3[x]
            myIndexVal3 = np.where(arr == np.amax(arr))
            # print(myIndexVal[0])
            myIndex3.append(myIndexVal3[0][0])
        # print(myIndex)
        myIndex4 = []
        for x in range(questions):
            arr = myPixelVal4[x]
            myIndexVal4 = np.where(arr == np.amax(arr))
            # print(myIndexVal[0])
            myIndex4.append(myIndexVal4[0][0])
        # print(myIndex)

    #     myIndex = myIndex1 + myIndex2 + myIndex3 + myIndex4

        # FINDING INDEX VALUES OF THE MARKINGS of ROLL
        myPixelValRoll = np.transpose(myPixelValRoll)
        myIndexRoll = []
        for x in range(7):
            arr = myPixelValRoll[x]
            myIndexValRoll = np.where(arr == np.amax(arr))
            # print(myIndexValRoll[0])
            myIndexRoll.append(myIndexValRoll[0][0])
        # print(myIndexRoll)
        global rollNo
        rollNo = int(''.join(map(str, myIndexRoll)))

        # GRADING
        grading1 = []
        for x in range(questions):
            if ans1[x] == myIndex1[x]:
                grading1.append(1)
            else:
                grading1.append(0)
        print(grading1)

        grading2 = []
        for x in range(questions):
            if ans2[x] == myIndex2[x]:
                grading2.append(1)
            else:
                grading2.append(0)
        print(grading2)

        grading3 = []
        for x in range(questions):
            if ans3[x] == myIndex3[x]:
                grading3.append(1)
            else:
                grading3.append(0)
        print(grading3)

        grading4 = []
        for x in range(questions):
            if ans4[x] == myIndex4[x]:
                grading4.append(1)
            else:
                grading4.append(0)
        print(grading4)

        grading = grading1 + grading2 + grading3 + grading4
        global score
        score = (sum(grading) / (questions*4)) * 100  # FINAL GRADE
        # print(score)

        # DISPLAYING ANSWERS
        imgResult1 = imgWarpColored1.copy()
        imgResult1 = utils.showAnswers(imgResult1, questions, choices, myIndex1, grading1, ans1)
        imgResult2 = imgWarpColored2.copy()
        imgResult2 = utils.showAnswers(imgResult2, questions, choices, myIndex2, grading2, ans2)
        imgResult3 = imgWarpColored3.copy()
        imgResult3 = utils.showAnswers(imgResult3, questions, choices, myIndex3, grading3, ans3)
        imgResult4 = imgWarpColored4.copy()
        imgResult4 = utils.showAnswers(imgResult4, questions, choices, myIndex4, grading4, ans4)

        imgRawDrawing1 = np.zeros_like(imgWarpColored1)
        imgRawDrawing1 = utils.showAnswers(imgRawDrawing1, questions, choices, myIndex1, grading1, ans1)
        imgRawDrawing2 = np.zeros_like(imgWarpColored2)
        imgRawDrawing2 = utils.showAnswers(imgRawDrawing2, questions, choices, myIndex2, grading2, ans2)
        imgRawDrawing3 = np.zeros_like(imgWarpColored3)
        imgRawDrawing3 = utils.showAnswers(imgRawDrawing3, questions, choices, myIndex3, grading3, ans3)
        imgRawDrawing4 = np.zeros_like(imgWarpColored4)
        imgRawDrawing4 = utils.showAnswers(imgRawDrawing4, questions, choices, myIndex4, grading4, ans4)

        invMatrix1 = cv2.getPerspectiveTransform(p12, p11)
        imgInvWarp1 = cv2.warpPerspective(imgRawDrawing1, invMatrix1, (widthImg, heightImg))
        invMatrix2 = cv2.getPerspectiveTransform(p22, p21)
        imgInvWarp2 = cv2.warpPerspective(imgRawDrawing2, invMatrix2, (widthImg, heightImg))
        invMatrix3 = cv2.getPerspectiveTransform(p32, p31)
        imgInvWarp3 = cv2.warpPerspective(imgRawDrawing3, invMatrix3, (widthImg, heightImg))
        invMatrix4 = cv2.getPerspectiveTransform(p42, p41)
        imgInvWarp4 = cv2.warpPerspective(imgRawDrawing4, invMatrix4, (widthImg, heightImg))

        imgRawGrade = np.zeros_like(imgGradeDisplay)
        cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5)
        # cv2.imshow("Grade", imgRawGrade)
        InvMatrixG = cv2.getPerspectiveTransform(pg2, pg1)
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, InvMatrixG, (heightImg, widthImg))

        # cv2.imshow("Grade", imgInvGradeDisplay)

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp1, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp2, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp3, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp4, 1, 0)
        # imgFinal = cv2.add(imgFinal, imgInvWarp)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)
        # imgFinal = cv2.add(imgFinal, imgInvGradeDisplay)

    imgBlank = np.zeros_like(img)
    imageArray1 = ([img, imgGray, imgBlur, imgCanny],
                  [imgContours, imgBubbleContours, imgWarpColored1, imgThresh1],
                  [imgResult1, imgRawDrawing1, imgInvWarp1, imgFinal])
    imageArray2 = ([img, imgGray, imgBlur, imgCanny],
                  [imgContours, imgBubbleContours, imgWarpColored2, imgThresh2],
                  [imgResult2, imgRawDrawing2, imgInvWarp2, imgFinal])
    imageArray3 = ([img, imgGray, imgBlur, imgCanny],
                  [imgContours, imgBubbleContours, imgWarpColored3, imgThresh3],
                  [imgResult3, imgRawDrawing3, imgInvWarp3, imgFinal])
    imageArray4 = ([img, imgGray, imgBlur, imgCanny],
                  [imgContours, imgBubbleContours, imgWarpColored2, imgThresh4],
                  [imgResult4, imgRawDrawing4, imgInvWarp4, imgFinal])
    # labels = [["Original", "Gray", "Blur", "Canny"],
    #           ["Contours", "Biggest Contour", "Warp", "Threshold"],
    #           ["Result", "Raw Drawing", "Inverse Warp", "Final"]]
    # imgStacked = utils.stackImages(imageArray, 0.3, labels)
    imgStacked1 = utils.stackImages(imageArray1, 0.3)
    imgStacked2 = utils.stackImages(imageArray2, 0.3)
    imgStacked3 = utils.stackImages(imageArray3, 0.3)
    imgStacked4 = utils.stackImages(imageArray4, 0.3)

    cv2.imshow("Stacked Images1", imgStacked1)
    # cv2.imshow("Stacked Images2", imgStacked2)
    # cv2.imshow("Stacked Images3", imgStacked3)
    # cv2.imshow("Stacked Images4", imgStacked4)
    cv2.imshow("Final Result", imgFinal)
    cv2.waitKey(0)

    return imgFinal
