import cv2
import pyvirtualcam
import numpy as np

faceCascade = cv2.CascadeClassifier('test.xml')
video_capture = cv2.VideoCapture(0)
video_capture.set(3,1280)
video_capture.set(4,720)
fmt = pyvirtualcam.PixelFormat.BGR
with pyvirtualcam.Camera(width=1280, height=720, fps=25, fmt=fmt) as cam:
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.BORDER_DEFAULT)

        # cv2.imshow('my webcam', frame)
        cam.send(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # esc to quit
    video_capture.release()
    cv2.destroyAllWindows()