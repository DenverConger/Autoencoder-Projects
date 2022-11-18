#Limited is better than nothing.
import cv2
vidcap = cv2.VideoCapture('/home/denmann99/autoencoder/videos/bryan/VID_20221103_175452276.mp4')
success,image = vidcap.read()
count = 0
while success:
  image = cv2.flip(image, 0)

  
  # image = cv2.resize(image, dim)
  cv2.imwrite(f"data/frame{count}.jpg", image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  
print("Conversion Done")
  