import numpy as np
import cv2

cap = cv2.VideoCapture('tree.avi')

while(cap.isOpened()):
  ret, frame = cap.read()

  if not ret:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  cv2.imshow('frame',gray)
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
