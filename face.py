import cv2 as cv
import numpy as np
import matplotlib as plot
import os
from os import listdir
from os.path import isfile, join
import re

# Loading haar cascade
face_cascade = cv.CascadeClassifier('./frontface_template.xml')
eye_cascade = cv.CascadeClassifier('./eye_template.xml')


def train(path):
    """ Train the model with the dataset in the specified path

    Arguments:
        path {string} -- Path where image filenames will be read from

    Returns:
        dictionary {dict} -- A dictionary mapping the labels to integers
        model {*} -- The trained model
    """

    pattern = '([a-z]|[A-Z])*'
    labels = []
    faces = []
    dictionary = dict()

    for f in listdir(path):
        if isfile(join(path, f)):
            label = re.search(pattern, f).group(0)
            if dictionary.get(label) is None:
                dictionary[str(label)] = len(labels)
                labels.append(len(labels))
            else:
                labels.append(dictionary.get(label))

            faces.append(detect(path + '/' + f))

    model = cv.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    return model, dictionary


def detect(path, color=False, coo=False, crop=True, eye=False):
    """ Detect face(s) in image

    Arguments:
        path {string} -- Path of the image

    Keyword Arguments:
        crop {bool} -- When true, if a face is found, return a cropped image around the first face found
                       If false, return the image in its original size with a rectangle drew around the face
        color {bool} -- Return an rgb image
        coo {bool} -- Return the rectangle coordinates drew around the face along with the image
        eye {bool} -- Enable eyes detection

    Returns:
        {image} -- Black and white image cropped around the face
    """

    # Read open image and convert it in black and white
    frame = cv.imread(path)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Faces are the tuples of 4 numbers
    # x,y => upperleft corner coordinates of face
    # width(w) of rectangle in the face
    # height(h) of rectangle in the face
    # frame means the input image to the detector
    # 1.1 is the kernel size or size of image reduced when applying the detection
    # 5 is the number of neighbors after which we accept that is a faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Arguments => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness
    if not crop:
        for (x, y, w, h) in faces:
            if color:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Eye detection
        if eye is True:
            # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = gray[y:y+h, x:x+w]

            # eyes detection
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            # Draw rectangles over eyes
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey),
                             (ex+ew, ey+eh), (255, 0, 0), 2)

    if not color:
        frame = gray
    if crop:
        (x, y, w, h) = faces[0]
        frame = frame[y:y+h, x:x+w]
    if coo:
        return frame, (y, y+h, x, x+w)
    else:
        return frame


def predict(model, labelDict, path):
    """ Make a prediciton with specifdied image with the given trained model
        It'll show up the image with a rectangle around the face, the guessed label and the confidence value

    Arguments:
        model {*} -- Trained model
        labelDict {dict} -- Dictionary mapping labels with integers
        path {string} -- Image path used for the prediction
    """

    img_predict = detect(path,
                         crop=False, color=True, coo=True)

    prediction = model.predict(detect(path))

    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (0, 0, 255)
    lineType = 4

    text = list(labelDict.keys())[
        list(labelDict.values()).index(prediction[0])]

    cv.putText(img_predict[0], text,
               (img_predict[1][2], img_predict[1][0] - 20),
               font,
               fontScale,
               fontColor,
               lineType)

    cv.putText(img_predict[0], str(round(prediction[1], 1)) + '%',
               (img_predict[1][2], img_predict[1][1] + 60),
               font,
               fontScale,
               fontColor,
               lineType)

    cv.imshow("Face found", img_predict[0])
    cv.waitKey(0)
    cv.destroyAllWindows()


model, label = train('./training')
predict(model, label, './prediction/pitt2.jpg')
