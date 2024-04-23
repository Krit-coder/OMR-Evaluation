import cv2
import numpy as np
import utils

score = 0


def check(img, questions, choices):
    ################################
    # path = "6.jpg"
    widthImg, heightImg = 700, 700
    # questions = 10
    # choices = 5
    ans = [0, 2, 2, 1, 0, 2, 2, 4, 3, 1]
    ################################

    # PREPROCESSING
    # img = cv2.imread(path)
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgFinal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # Finding all contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5)

    # FIND RECTANGLES
    rectCont = utils.rectContour(contours)
    biggestContour = utils.getCornerPoints(rectCont[0])
    gradePoints = utils.getCornerPoints(rectCont[1])
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

        pg1 = np.float32(gradePoints)
        pg2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(pg1, pg2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
        # cv2.imshow("Grade", imgGradeDisplay)

        # APPLY THRESHOLD
        imgWarpGary = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGary, 170, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = utils.splitBoxes(imgThresh, questions, choices)
        # cv2.imshow("Boxes:", boxes[0])
        # cv2.imshow("Boxes:", boxes[2])

        # print(cv2.countNonZero(boxes[0]), cv2.countNonZero(boxes[1]), cv2.countNonZero(boxes[2]), cv2.countNonZero(
        #     boxes[3]), cv2.countNonZero(boxes[4]))

        # GETTING NON ZERO PIXEL VALUES OF EACH BOX
        myPixelVal = np.zeros((questions, choices))
        countC = 0
        countR = 0
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if countC == choices:
                countR += 1
                countC = 0

        # print(myPixelVal)

        # FINDING INDEX VALUES OF THE MARKINGS
        myIndex = []
        for x in range(questions):
            arr = myPixelVal[x]
            myIndexVal = np.where(arr == np.amax(arr))
            # print(myIndexVal[0])
            myIndex.append(myIndexVal[0][0])
        # print(myIndex)

        # GRADING
        grading = []
        for x in range(questions):
            if ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        # print(grading)
        global score
        score = (sum(grading) / questions) * 100  # FINAL GRADE
        # print(score)

        # DISPLAYING ANSWERS
        imgResult = imgWarpColored.copy()
        imgResult = utils.showAnswers(imgResult, questions, choices, myIndex, grading, ans)

        imgRawDrawing = np.zeros_like(imgWarpColored)
        imgRawDrawing = utils.showAnswers(imgRawDrawing, questions, choices, myIndex, grading, ans)

        invMatrix = cv2.getPerspectiveTransform(p2, p1)
        imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

        imgRawGrade = np.zeros_like(imgGradeDisplay)
        cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5)
        # cv2.imshow("Grade", imgRawGrade)
        InvMatrixG = cv2.getPerspectiveTransform(pg2, pg1)
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, InvMatrixG, (heightImg, widthImg))

        # cv2.imshow("Grade", imgInvGradeDisplay)

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        # imgFinal = cv2.add(imgFinal, imgInvWarp)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)
        # imgFinal = cv2.add(imgFinal, imgInvGradeDisplay)

    imgBlank = np.zeros_like(img)
    imageArray = ([img, imgGray, imgBlur, imgCanny],
                  [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
                  [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
    # labels = [["Original", "Gray", "Blur", "Canny"],
    #           ["Contours", "Biggest Contour", "Warp", "Threshold"],
    #           ["Result", "Raw Drawing", "Inverse Warp", "Final"]]
    # imgStacked = utils.stackImages(imageArray, 0.3, labels)
    imgStacked = utils.stackImages(imageArray, 0.3)

    # cv2.imshow("Final Result", imgFinal)
    # cv2.imshow("Stacked Images", imgStacked)
    cv2.waitKey(0)

    return imgFinal
