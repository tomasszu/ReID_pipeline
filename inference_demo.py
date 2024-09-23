import os
import matplotlib.pyplot as plt
from PIL import Image
import misc.fisheye_panorama as toPanorama
import misc.detect_and_crop as detectAndCrop
import vehicle_ReID_croppedData as ReID

fisheye_folder = 'source_images/fisheye_test/'
fisheye_image = '01_fisheye_day_000000.jpg'

#Convert fisheye photo to panorama
toPanorama.fisheye_to_pano(fisheye_folder,fisheye_image)

os.chdir('/home/tomass/tomass/ReID_pipele')

pano_folder = 'source_images/panoramas/'
pano_image = 'panorama_01_fisheye_day_000024.jpg'

#Detect vehicles and crop
detectAndCrop.crop_detected_objects(pano_folder,pano_image)

os.chdir('/home/tomass/tomass/ReID_pipele')

im_number = 5
query_folder = "panoramas/panorama_01_fisheye_day_000000.jpg/"
gallery_folder = "panoramas/panorama_01_fisheye_day_000024.jpg/"

ReID.visualise_main(im_number, query_folder, gallery_folder)