# Vehicle ReID pipeline. (Object detection -> cropped object image -> object ReID in other images via feature extraction model from https://github.com/regob/vehicle_reid

The project is using a modified version of the vehicle_reid model avilable from the regob Github repo:
https://github.com/regob/vehicle_reid

This model is located in the vehicle_reid_repo2/vehicle_reid folder.

<h2>Training a model</h2>

To train the model, the vehicle_reid_repo2/vehicle_reid/train.py can be used, mostly by following the instructions from the original repo. In this case only the links to the data need to be changed and a folder path vehicle_reid_repo/vehicle_reid/automated_training/ needs to be provisioned, where .txt files with information about training loss and accuracy are saved.

<h2>Testing a ReID model</h2>

One of the ways to test a ReID model is through the test scenario with videos from the AICity challenge. This scenario has been tailored specifically to ReID vehicles from AI_City_01_Itersection vdo4.avi to vdo1.avi.

This code is available at ground_truth_ReID_test.py . The test scenario leverages the ground truth data available from the AI City challenge datasets and enables testing of ReID without need for detecting the vehicles in the images first, since we are provided with the coordinates of their bounding boxes and their assigned Id's.

Variables like video_path_1, video_path_2, ground_truths_path_1 ... need to lead to the appropriate AICity files.

The ground_truth_ReID_test.py uses the save_extractions_to_lance_db and compare_extractions_to_lance_db functions from the fExtract import file. In the upper part of the file the appropriate import is selected, based on the type of model you want to use. 

The default import is counting_workspace.misc.feature_extract_AICity as fExtract to use any model trained form the included train repo that outputs a 512 long vector, as provisioned in the [lance_db_init](counting_workspace/misc/lance_db_init.py) file.