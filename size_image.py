import cv2 
import os 

folder_dir = "img/"
for image in os.listdir(folder_dir):
    if image.endswith(".jpg") or image.endswith("PNG"):
        img = cv2.imread(folder_dir + image) 
        width, height = img.shape[:2]
        print(f"{image}: height: {height} width: {width}")
