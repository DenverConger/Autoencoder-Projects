from pathlib import Path
from os import sys
import matplotlib.pyplot as plt
import PIL
sys.path.insert(0, Path.cwd().parent)
import cv2
from autocrop import Cropper

from glob import glob

faces = [f for f in glob("data/uchtdorf/*") if not f.endswith("md")]
print(f"{len(faces)} images to test with.")

c = Cropper(width=512, height=512, face_percent=40)


def save_crops(faces, cropper):
    """Given a list on filepaths, crops and plots them."""
    count = 0
    for face in faces:
        try:
            img_array = c.crop(face)
        except (AttributeError, TypeError):
            pass
        if img_array is not None:
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"pics/uchtdorf/uchtdorf_face{count}.jpg", img_array_rgb)
            print(f"saved to pics/uchtdorf/uchtdorf_face{count}.jpg")
        count +=1
          
save_crops(faces, c)