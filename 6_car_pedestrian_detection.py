import cv2
import numpy as np

# classifier object dalam sebuah video dengan menggunakan haar cascade
# xml classifier ny sudah disediakan dari cv2 nya, kita tinggal pake

# body classifier for car
# body_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# cap for cars
# cap = cv2.VideoCapture('cars.avi')
cap = cv2.VideoCapture('walking.avi')

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect object yang ada dalam scene dengan menggunakan haarcascade
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        cv2.imshow('Pedestrians', frame)
    
    if(cv2.waitKey(1) == 13):
        break