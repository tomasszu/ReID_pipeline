import os
import cv2
from PIL import Image

sequence_folder = "/home/tomass/Downloads/20140618_Sequence1a/Sequence1a/KAB_SK_4_undist_converted"
sequence_images = sorted(os.listdir(sequence_folder))

os.chdir(sequence_folder)

img_array = []

for x in sequence_images: #can base on number of image in your directory
    print("adding: " + x) # use this to append your image into array

    img = cv2.imread(x)
    img_array.append(img)

#Create Video
out = cv2.VideoWriter('KAB_SK_4_undist.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (img_array[1].shape[1], img_array[1].shape[0]))
for i in range(len(img_array)):
    out.write(img_array[i])
    #cv2.imshow("frame", img_array[i])
    #cv2.waitKey(0)
    print("writing: " + str(i))
out.release()