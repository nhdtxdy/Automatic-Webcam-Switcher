import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

video=cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)
segmentor = FaceDetector()

fpsReader = cvzone.FPS()

while True:
    ret, frame = video.read()
    imgOut, bboxes = segmentor.findFaces(frame, draw=True)

    imgStack = cvzone.stackImages([imgOut], 2, 1)
    _, imgStack = fpsReader.update(imgStack, color=(0,0,255))

    cv2.imshow('Image', imgStack)
    cv2.waitKey(1)