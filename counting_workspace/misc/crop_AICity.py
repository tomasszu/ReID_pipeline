import cv2
import os
import sys


sequence_name = "AIC22_Track1_MTMC_train_S01"

def crop_from_bbox(frame, track_id_name, xyxy, folder):
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    cropped_img = frame[y1:y2, x1:x2]
    #cv2.imshow('image',cropped_img)

    # if(folder.__contains__('1')):
    #     print("crop debug:")
    #     print(f"{track_id_name} : {xyxy}")

    file_name = os.path.join(sys.path[0], f'cropped/{sequence_name}/{folder}/' + f'id_{track_id_name}.jpg')
    print(file_name)
    # os.chdir(save_dir)
    # print(os.listdir(save_dir))
    # print(os.getcwd())
    if(os.path.exists(file_name) is False):
        if(min(x1, y1, x2, y2) > 0):
            cv2.imwrite(file_name, cropped_img)
    #cv2.waitKey(0)