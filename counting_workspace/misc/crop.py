import cv2
import os


def crop_from_bbox(frame, track_id_name, xyxy):
    x1, y1, x2, y2 = xyxy[0].astype("int"), xyxy[1].astype("int"), xyxy[2].astype("int"), xyxy[3].astype("int")
    cropped_img = frame[y1:y2, x1:x2]
    #cv2.imshow('image',cropped_img)

    intersection = "intersection_1"

    save_dir = f'../cropped/{intersection}/'
    # os.chdir(save_dir)
    # print(os.listdir(save_dir))
    # print(os.getcwd())
    print(cv2.imwrite(f'id_{track_id_name}.jpg', cropped_img))
    #cv2.waitKey(0)