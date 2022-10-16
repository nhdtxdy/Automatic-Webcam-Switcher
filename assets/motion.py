#!/usr/bin/python

import sys
import cv2
import numpy as np
import time

if len(sys.argv) < 2:
    video_capture = cv2.VideoCapture(0)
else:
    filepath = sys.argv[1]
    video_capture = cv2.VideoCapture(filepath)

# Read two frames, last and current, and convert current to gray.
ret, last_frame = video_capture.read()
ret, current_frame = video_capture.read()
gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

i = 0
while True:

    last_frame = current_frame
    ret, current_frame = video_capture.read()

    if not ret:
        break

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(last_frame, current_frame)
    '''
    i += 1
    if i == 10:
        i = 0
        print("Current frame = ", np.mean(current_frame))
        print("Diff = ", np.mean(diff))
    '''

    if np.mean(diff) > 1:
        i += 1
        print("Achtung! Motion detected", i)

    cv2.imshow("Motion detected", diff)
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()