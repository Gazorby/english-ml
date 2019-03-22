import cv2 as cv
import numpy as np
import matplotlib as plot
import os

# Loading haar cascade
face_cascade = cv.CascadeClassifier('./frontface_template.xml')
eye_cascade = cv.CascadeClassifier('./eye_template.xml')
smile_cascade = cv.CascadeClassifier('./smile_template.xml')
lowerbody_cascade = cv.CascadeClassifier('./lowerbody_template.xml')
fullbody_cascade = cv.CascadeClassifier('./fullbody_template.xml')
upperbody_cascade = cv.CascadeClassifier('./upperbody_template.xml')


def detect(imgPath, eye=True, smile=False, fullBody=False, upperBody=False, lowerBody=False):
    # Read open image and convert it in black and white
    frame = cv.imread(imgPath, cv.COLOR_BGR2GRAY)

    # Faces are the tuples of 4 numbers
    # x,y => upperleft corner coordinates of face
    # width(w) of rectangle in the face
    # height(h) of rectangle in the face
    # frame means the input image to the detector
    # 1.1 is the kernel size or size of image reduced when applying the detection
    # 5 is the number of neighbors after which we accept that is a faces
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5)

    # Arguments => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Smile detection
        if smile is True:
            smiles = smile_cascade.detectMultiScale(frame)
            for (sx, sy, sw, sh) in smiles:
                # Draw rectangles over faces
                cv.rectangle(frame, (sx, sy),
                             (sx+sw, sy+sh), (0, 0, 255), 2)
        # Eye detection
        if eye is True:
            # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
            roi_gray = frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # eyes detection
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            # Draw rectangles over eyes
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey),
                             (ex+ew, ey+eh), (255, 0, 0), 2)
    return frame


frame = detect('./training/gari.jpg')

cv.imshow("Faces found", frame)
cv.waitKey(0)
cv.destroyAllWindows()
