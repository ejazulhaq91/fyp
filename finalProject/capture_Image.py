import cv2 as cv
import numpy as np
import os


def take_image():
    name = input("Enter Name of Student: ")
    os.chdir('faces/')
    os.mkdir(name)
    capture_Images = cv.VideoCapture(0)
    sampleCounter = 1
    while True:
        isFrame, frame = capture_Images.read()
        cv.imwrite(name + '/' + name + "-" + str(sampleCounter) + ".jpg", frame)
        cv.putText(frame, f"Sample No: {sampleCounter}", (frame.shape[0] // 2 - 10, 25),
                   cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
        cv.imshow("Taking Images", frame)
        cv.waitKey(200)
        if sampleCounter > 49:
            break
        sampleCounter += 1

    capture_Images.release()
    cv.destroyAllWindows()


take_image()
