import math
import os
import cv2
import numpy as np


def fisheye_to_pano(img_folder, image_dir):
       # Load the fisheye image
       read_dir = img_folder
       im_dir = image_dir
       image = cv2.imread(read_dir + im_dir)

       width = int((image.shape[0])/2)
       #height = image.shape[1]

       print(width)

       panorama = np.zeros((width, 4*width, 3), np.uint8)



       for i in range(width):
              for j in range(4*width):
                     radius = width - i
                     #theta = (2*math.pi*(j)/(4*width))-5.2
                     theta = (2*math.pi*(j)/(4*width))

                     x = width - int(round(radius*math.cos(theta)))
                     y = width - int(round(radius*math.sin(theta)))                

                     if(x >= 0 and x < 2*width and y >= 0 and y < 2*width):
                            panorama[i][j] = image[x][y]



       save_dir = f'source_images/panoramas/'

       if os.path.exists(save_dir):
              os.chdir(save_dir)
       else:
              os.makedirs(save_dir)
              os.chdir(save_dir)

       # Save the undistorted image
       cv2.imwrite(f'panorama_{im_dir}', panorama)


       # Display the panorama image
       #cv2.imshow('Panorama Image', panorama)
       #cv2.waitKey(0)
       cv2.destroyAllWindows()
