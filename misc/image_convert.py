from PIL import Image
from os import listdir, path, remove

image_path = '/home/tomass/Downloads/20140618_Sequence1a/Sequence1a/KAB_SK_1_undist_converted/'
images= listdir(image_path)
for img in images:
    file_name, file_type = path.splitext(img)
    if file_type not in ['.py','.jpg']:
        im = Image.open(image_path + img)
        print(path.join(image_path + str(img).replace(".bmp",".jpg")))
        im.save(path.join(image_path+ str(img).replace(".bmp",".jpg")))
        remove(path.join(image_path + str(img)))