from wand.image import Image
import numpy as np
import cv2


with Image(filename='/home/tomass/Downloads/Traffic_vehicles_Object_Detection/10260093.jpg') as img:
    print(img.size)
    img.virtual_pixel = 'transparent'
    #img.distort('polar', (500.0, 0.0, 500.0, 300.0)) #(0.2, 0.0, 0.0, 1.0)
    img.distort('sentinel', (0.2, 0.0, 0.1, 0.7)) #(0.2, 0.0, 0.0, 1.0)
    img.save(filename='checks_barrel.png')
    # convert to opencv/numpy array format
    img_opencv = np.array(img)

# display result with opencv
cv2.imshow("BARREL", img_opencv)
cv2.waitKey(0)