import sys
import cv2
import pyvirtualcam
import numpy as np
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

def webcam_face_detect(video_mode):
    video_capture = cv2.VideoCapture(video_mode)
    video_capture.set(3,1280)
    video_capture.set(4,720)
    num_faces = 0

    brb = cv2.imread('brb.jpg')
    brb = cv2.flip(brb, 1)

    fmt = pyvirtualcam.PixelFormat.BGR
    cnt = 0
    ret, last_frame = video_capture.read()
    ret, frame = video_capture.read()
    faceDetector = FaceDetector()
    with pyvirtualcam.Camera(width=1280, height=720, fps=60, fmt=fmt) as cam:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            imgOut, bboxes = faceDetector.findFaces(frame, draw=False)

            # eyes = eye_cascade.detectMultiScale(gray)
            if len(bboxes):
                cnt = 0
                cam.send(cv2.flip(frame, 1))
            else:
                cnt += 1
                if cnt >= 20:
                    cam.send(brb)
                else:
                    cam.send(cv2.flip(frame, 1))

    video_capture.release()
    cv2.destroyAllWindows()
    return num_faces


if __name__ == "__main__":
    if len(sys.argv) < 2:
        video_mode= 0
    else:
        video_mode = sys.argv[1]

    webcam_face_detect(video_mode)