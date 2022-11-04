from pathlib import Path
from os import sys
import matplotlib.pyplot as plt
import PIL
sys.path.insert(0, Path.cwd().parent)
import cv2
from autocrop import Cropper

from glob import glob

faces = [f for f in glob("data/*") if not f.endswith("md")]
print(f"{len(faces)} images to test with.")

c = Cropper(face_percent=50)


count = 0
for face in faces:
    face = cv2.imread(face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    try:
        img_array = c.crop(face)
    except (AttributeError, TypeError):
        pass
    if img_array is not None:
        cv2.imwrite(f"pics/denver/denver_face{count}.jpg", img_array)
        print(f"saved to pics/denver/denver_face{count}.jpg")
    count +=1