#Limited is better than nothing.
import cv2
vidcap = cv2.VideoCapture('videos/uchtdorf/2014-10-3040-president-dieter-f-uchtdorf-720p-eng.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("data/uchtdorf/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  