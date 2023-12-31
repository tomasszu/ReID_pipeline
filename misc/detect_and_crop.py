import torch
import cv2
import os


def crop_detected_objects(pano_folder, pano_name):

    needed_classes = ['car', 'bus', 'truck']


    #Yolo Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


    im_folder = pano_folder
    im_name = pano_name

    im = im_folder + im_name

    results = model(im)

    print(results)

    #print(results.pandas().xyxy[0]['xmin'])

    predictions = results.pandas().xyxy[0]

    xmins, ymins, xmaxs, ymaxs, confidences, classes = predictions['xmin'].astype("int"), predictions['ymin'].astype("int"), predictions['xmax'].astype("int"), predictions['ymax'].astype("int"), predictions['confidence'], predictions['name']

    p_img = cv2.imread(im)

    # x1, y1, x2, y2 = xmins[1], ymins[1], xmaxs[1], ymaxs[1]

    # p_img = p_img[y1:y2, x1:x2]

    save_dir = f'../cropped/{im_folder}/{im_name}'

    if os.path.exists(save_dir):
        os.chdir(save_dir)
    else:
        os.makedirs(save_dir)
        os.chdir(save_dir)


    #Iterating through detections and saving cropped images
    for i in range(classes.size):
    #     print(i)
        class_name = classes[i]
        score = round(confidences[i], 2)

        if(needed_classes.__contains__(class_name) and score > 0.3):
            x1, y1, x2, y2 = xmins[i], ymins[i], xmaxs[i], ymaxs[i]
            cropped_img = p_img[y1:y2, x1:x2]
            #cv2.imshow('image',cropped_img)
            cv2.imwrite(f'im{i}_{class_name}.jpg', cropped_img)
            #cv2.waitKey(0)


    #if out of memory issues arise, we come here to cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    #cv2.destroyAllWindows()
    #         #pass

    # cv2.imshow('image',cropped_img)

    # cv2.waitKey(0)

