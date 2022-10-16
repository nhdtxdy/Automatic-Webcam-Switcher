import sys
import cv2
import pyvirtualcam
import numpy as np
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

def webcam_face_detect(video_mode, src):
    video_capture = cv2.VideoCapture(video_mode)    
    video_player = cv2.VideoCapture(src)

    video_capture.set(3, 1280)
    video_capture.set(4, 720)

    video_player.set(3, 1280)
    video_player.set(4, 720)

    num_faces = 0

    brb = cv2.imread('brb.jpg')
    brb = cv2.flip(brb, 1)

    fmt = pyvirtualcam.PixelFormat.BGR
    cnt = 0
    ret, last_frame = video_capture.read()
    ret, frame = video_capture.read()
    faceDetector = FaceDetector()
    needReset = True
    with pyvirtualcam.Camera(width=1280, height=720, fps=30, fmt=fmt) as cam:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            imgOut, bboxes = faceDetector.findFaces(frame, draw=False)

            # eyes = eye_cascade.detectMultiScale(gray)

            frame = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

            if len(bboxes):
                cam.send(frame)
                needReset = True
            else:
                if needReset:
                    video_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = video_player.read()
                if not ret2:
                    cam.send(frame)
                    video_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: 
                    frame2 = cv2.resize(frame2,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    cam.send(frame2)
                    needReset = False

    video_capture.release()
    cv2.destroyAllWindows()
    return num_faces


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 3:
        video_mode= 0
    else:
        video_mode = sys.argv[2]

    webcam_face_detect(video_mode, sys.argv[1])