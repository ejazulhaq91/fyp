import cv2 as cv
import os
import numpy as np
from mtcnn.mtcnn import MTCNN


def detect_face():
    name = input("Enter Name of Student: ")
    os.chdir('faces/')
    os.mkdir(name)
    face_detector = MTCNN()
    capture_video = cv.VideoCapture(0)
    sampleCounter = 1
    while True:
        frame_got, frame = capture_video.read()
        detected_faces = face_detector.detect_faces(frame)
        if detected_faces:
            for person in detected_faces:
                boxes = person['box']
                cv.rectangle(frame, (boxes[0], boxes[1]), (boxes[0] + boxes[2],
                                                           boxes[1] + boxes[3]), (0, 255, 0), 2)
        cv.imwrite(name + '/' + name + "-" + str(sampleCounter) + ".jpg", frame)
        cv.putText(frame, f"Sample No: {sampleCounter}", (frame.shape[0] // 2 - 10, 25),
                   cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
        cv.imshow("Frame", frame)
        sampleCounter += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture_video.release()
    cv.destroyAllWindows()


detect_face()
