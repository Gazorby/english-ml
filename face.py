import cv2 as cv
import numpy as np
import matplotlib as plot
import os

face_cascade = cv.CascadeClassifier('./frontface_template.xml')
eye_cascade = cv.CascadeClassifier('./eye_template.xml')


def detectFace(imgPath, eye=True):
    frame = cv.imread(imgPath, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if eye is True:
            roi_gray = frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey),
                             (ex+ew, ey+eh), (255, 0, 0), 2)
    return frame


frame = detectFace('./training/gari.jpg')

cv.imshow("Faces found", frame)
cv.waitKey(0)
cv.destroyAllWindows()
