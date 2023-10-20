import os
import cv2
from PIL import Image

sequence_folder = "/home/tomass/Downloads/VIP_CUP_2020/fisheye_video_1"
sequence_images = sorted(os.listdir(sequence_folder))

os.chdir(sequence_folder)

img_array = []

for x in sequence_images: #can base on number of image in your directory
    print("adding: " + x) # use this to append your image into array

    img = cv2.imread(x)
    img_array.append(img)

#Create Video
out = cv2.VideoWriter('fisheye_vid_1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 5, (img_array[0].shape[0], img_array[0].shape[1]))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()