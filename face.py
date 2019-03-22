import cv2 as cv
import numpy as np
import matplotlib as plot
import os


def init(imgDir):
    imgList = os.listdir(imgDir)
    imgMat = []
    for img in imgList:
        tmp = cv.imread('./training/' + img)
        imgMat.append(cv.cvtColor(tmp, cv.COLOR_BGR2GRAY))
    return imgMat


images = cv.imread('./training/gari2.jpeg')
gray = cv.cvtColor(images, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier('./frontface_template.xml')
faces = face_cascade.detectMultiScale(images,
                                      scaleFactor=1.1,
                                      minNeighbors=5)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv.rectangle(images, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # roi_gray = images[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]

cv.imshow("Faces found", images)
cv.waitKey(0)
cv.destroyAllWindows()
