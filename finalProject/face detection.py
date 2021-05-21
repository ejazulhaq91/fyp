# Using HaarCascade Method

# import cv2 as cv
# import os
#
#
# def capture_image():
#     name = input("Enter Name of Student: ")
#     os.chdir('faces/')
#     os.mkdir(name)
#
#     face_detector = cv.CascadeClassifier("C:\\Users\\user\\PycharmProjects\\finalProject\\haarcascade_frontalface_default.xml")
#     capture = cv.VideoCapture(0)
#     imgSample = 0
#
#     while True:
#         ret, frame = capture.read()
#         gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         #cv.imshow('Image', frame)
#
#         detected_faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)
#         for x, y, w, h in detected_faces:
#             cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv.imshow('Image', frame)
#         if cv.waitKey(1) & 0xFF == ord('k'):
#             cv.imwrite(name + '/' + name+"-"+str(imgSample)+".jpg", frame[y:y+h, x:x+w])
#             imgSample += 1
#         elif cv.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     capture.release()
#     cv.destroyAllWindows()
#
# capture_image()

# Using deep learning Face Recognition

# import cv2 as cv
# import face_recognition
#
# video = cv.VideoCapture(0)
#
# face_locations = []
#
# while True:
#     ret, frame = video.read()
#     rgb_frame = frame[:,:,::-1]
#     face_locations = face_recognition.face_locations((rgb_frame))
#     for top, right, bottom, left in face_locations:
#         cv.rectangle(frame, (top, left), (right, bottom), (0,0,255), 3)
#         cv.imshow('Video',frame)
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#
# video.release()
# cv.destroyAllWindows()






# import requests
# import cv2
# import numpy as np
# import imutils
#
# # Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
# url = "http://10.5.87.93:8080/shot.jpg"
#
# # While loop to continuously fetching data from the Url
# while True:
#     img_resp = requests.get(url)
#     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     img = cv2.imdecode(img_arr, -1)
#     img = imutils.resize(img, width=1000, height=1800)
#     cv2.imshow("Android_cam", img)
#
#     # Press Esc key to exit
#     if cv2.waitKey(1) == 27:
#         break
#
# cv2.destroyAllWindows()