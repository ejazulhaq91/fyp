import cv2 as cv
import os
import numpy as np
from mtcnn.mtcnn import MTCNN


def detecting_faces():
    face_detector = MTCNN()
    capture_video = cv.VideoCapture(0)
    cropped_faces = []
    while True:
        frame_got, frame = capture_video.read()
        detected_faces = face_detector.detect_faces(frame)
        if detected_faces:
            for person in detected_faces:
                boxes = person['box']
                cv.rectangle(frame, (boxes[0], boxes[1]), (boxes[0] + boxes[2],
                                                           boxes[1] + boxes[3]), (0, 255, 0), 2)
                cropped_faces = frame[boxes[0] - 10:boxes[0] + boxes[2] + 50, boxes[1] - 10:boxes[1] + boxes[3] + 50]
        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture_video.release()
    cv.destroyAllWindows()
    return cropped_faces


face_detected = detecting_faces()
print(face_detected)
