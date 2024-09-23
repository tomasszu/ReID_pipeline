#Trennējot modelim iziet cauri:
#   warm_epoch ?? (5%-20% jeb 0-4 epohi šajaa gad)
#   lr (0.01 - 0.1 , jeb 0.01;0,025;0.05;0,075;0.1)
#   fp16 (Ir vai nav) 
#   cosine (learning_rate) (Ir vai nav)
#   stride ?? (1,2,3) ?? Ja var vsp to savietot ar konkretu output dimension
#   droprate (Parasti 0.1-0.5, pameginasim 0,1;0,3;0,4;0.5;0.6)
#   erasing_p (Parasti starp 0.3-0.7 , pameginasim 0.3;0.5;0.7)
#   color_jitter ?? (True/ false vai tieši 0 - 1)
#   label_smoothing ?? (between 0.1 and 0.3, by default -0.0, p-ameginasim 0.0;0.1;0.2)
#   model - (resnet_ibn, resnet, densenet, swin, efficientnet) 
#   loss(triplet/contrast/instance/arcface/cosface/circle/sphere)
#   linear_num (default 512, pameginasim 256, 512, 1024)

#Additionally uzstādīt pareizo:
#   name
#   

# import os
# import argparse
# import torch
# import torchvision
# import warnings

# from train_modified_for_automation import train_model

import subprocess
import csv

import json

#Number of chosen options per parameter

num_warm_epoch = 5
num_lr = 5
num_cos_lr = 2
#num_fp16 = 2
num_stride = 3
num_droprate = 5
num_erasing_prob = 3
num_color_jitter = 2
num_label_smoothing = 3
num_model = 5
num_loss = 7
num_linear_num = 3


#Lists of options per parameter

opt_warm_epoch = [0,1,2,3,4]
opt_lr = [0.01,0.025,0.05,0.075,0.1]
opt_cos_lr = [False, True]
#opt_fp16 = [True, False]
opt_stride = [1,2,3]
opt_droprate = [0.1,0.3,0.4,0.5,0.6]
opt_erasing_prob = [0.3,0.5,0.7]
opt_color_jitter = [False, True]
opt_label_smoothing = [0.0,0.1,0.2]
opt_model = ["resnet_ibn", "resnet", "densenet", "swin", "efficientnet"]
opt_loss = ["triplet", "contrast", "instance", "arcface", "cosface", "circle", "sphere"]
opt_linear_num = [256, 512, 1024]

#Sequence of the testing matrix

num_epochs = 6 #Ideja ka 2x mazaak mes liekam masiivaa
opt_epochs = ["5","7","9","11","13","15"] # Jaskataas vai ar to save_freq parametra nomaiņu pietika lai šitā sanāktu


# testing_matrix = [num_model][num_loss][num_label_smoothing][num_erasing_prob][num_droprate][num_lr][num_warm_epoch][num_stride][num_linear_num][num_color_jitter][num_cos_lr][num_epochs][num_modes]

# #Value piešķiršana
# for st, stride in enumerate(opt_stride):
#     for ln, linear_num in enumerate(opt_linear_num):
#         for jt, jitter in enumerate(opt_color_jitter):
#             for cl, cos_lr in enumerate(opt_cos_lr):
#                 for ep, epoch in enumerate(opt_epochs): # Te arī vajag lai viss atbilst
#                     for md, mode in enumerate(opt_modes):
#                         testing_matrix[0][0][0][0][0][0][0][st][ln][jt][cl][ep][md] = "0.01,0.06"

# TRAINING MATRIX 1. DAĻA -------------------------------------------------------------------------

# Path to the Python file you want to run
train_file_path = "vehicle_reid_repo/vehicle_reid/train_modified_for_automation.py"

#results_matrix = [num_stride][num_linear_num][num_color_jitter][num_cos_lr][num_epochs]
results_matrix = [[[[[0 for _ in range(num_epochs)] for _ in range(num_cos_lr)] for _ in range(num_color_jitter)] for _ in range(num_linear_num)] for _ in range(num_stride)]

# # Initialize a nested dictionary to hold values for JSON
json_data = {}

# #Trennesana
for st, stride in enumerate(opt_stride):
    json_data[str(stride)] = {}
    for ln, linear_num in enumerate(opt_linear_num):
        json_data[str(stride)][str(linear_num)] = {}
        for jt, jitter in enumerate(opt_color_jitter):
            json_data[str(stride)][str(linear_num)][str(jitter)] = {}
            for cl, cos_lr in enumerate(opt_cos_lr):
                json_data[str(stride)][str(linear_num)][str(jitter)][str(cos_lr)] = {}
                #TRAIN
                # Run the Python file and wait for its execution to finish
                # #result = subprocess.run(["python3", python_file_path, "--name=test_result", "--warm_epoch=1", "--lr=0.1", "--cosine", "--fp16", "--stride=1", "--droprate=0.1", "--erasing_p=0.1", "--color_jitter", "--label_smoothing=0.1"], capture_output=False)

                if jitter is True:
                    if cos_lr is True:
                        result = subprocess.run(["python3", train_file_path, f"--name=train1_{st}{ln}{jt}{cl}_output","--total_epoch=16", "--fp16", f"--stride={stride}", f"--linear_num={linear_num}", "--color_jitter", "--cosine"], capture_output=False)
                    else:
                        result = subprocess.run(["python3", train_file_path, f"--name=train1_{st}{ln}{jt}{cl}_output","--total_epoch=16", "--fp16", f"--stride={stride}", f"--linear_num={linear_num}", "--color_jitter"], capture_output=False)
                else:
                    if cos_lr is True:
                        result = subprocess.run(["python3", train_file_path, f"--name=train1_{st}{ln}{jt}{cl}_output","--total_epoch=16", "--fp16", f"--stride={stride}", f"--linear_num={linear_num}", "--cosine"], capture_output=False)
                    else:
                        result = subprocess.run(["python3", train_file_path, f"--name=train1_{st}{ln}{jt}{cl}_output","--total_epoch=16", "--fp16", f"--stride={stride}", f"--linear_num={linear_num}"], capture_output=False)
                # Check if the execution was successful
                if result.returncode == 0:
                    print("Training Script executed successfully!")
                    print(result)
                else:
                    print("An error occurred while executing the script.")
                    print("Error output:", result.stderr.decode())
                with open(f"vehicle_reid_repo/vehicle_reid/automated_training/train1_{st}{ln}{jt}{cl}_output.txt", "r") as file:
                    for line in file:
                        res_strings = line.split(',')
                        if res_strings[0] in opt_epochs:
                            state = str(res_strings[0] + " " + res_strings[1])
                            value = res_strings[2],res_strings[3].rstrip('\n')
                            json_data[str(stride)][str(linear_num)][str(jitter)][str(cos_lr)][state] = value
                            # Write the JSON data to a file
                            with open("vehicle_reid_repo/vehicle_reid/automated_training/train1_output.json", "w") as json_file:
                                json.dump(json_data, json_file, indent=4)
                        #testing_matrix[st][ln][jt][cl][ep] = "0.01,0.06"


#-------------------------------------------------------------------------------------------------

# #Value piešķiršana
# for st, stride in enumerate(opt_stride):
#     for ln, linear_num in enumerate(opt_linear_num):
#         for jt, jitter in enumerate(opt_color_jitter):
#             for cl, cos_lr in enumerate(opt_cos_lr):
#                 for ep, epoch in enumerate(opt_epochs): # Te arī vajag lai viss atbilst
#                     for md, mode in enumerate(opt_modes):
#                         testing_matrix[0][0][0][0][0][0][0][st][ln][jt][cl][ep][md] = "0.01,0.06"


####################### Simulation code: ###########################################

# opt_upper = ["one", "two", "three"]
# opt_middle = ["viens", "divi", "tris"]
# opt_lower = [100, 200, 300]


# # Define the dimensions
# rows = opt_upper.__len__()
# cols = opt_middle.__len__()
# depth = opt_lower.__len__()

# num_epochs = 8 #Ideja ka 2x mazaak mes liekam masiivaa
# opt_epochs = ["1","3","5","7","9","11","13","15"] # Jaskataas vai ar to save_freq parametra nomaiņu pietika lai šitā sanāktu


# # Initialize the multidimensional array with zeros
# json_matrix = [[[[0 for _ in range(num_epochs)] for _ in range(depth)] for _ in range(cols)] for _ in range(rows)]
# # print(json_matrix)

# # Initialize a nested dictionary to hold values for JSON
# json_data = {}

# # Value assignment and JSON creation simultaneously
# for u, upp in enumerate(opt_upper):
#     json_data[upp] = {}
#     for m, mid in enumerate(opt_middle):
#         json_data[upp][mid] = {}
#         for l, low in enumerate(opt_lower):
#             json_data[upp][mid][str(low)] = {}
#             with open("vehicle_reid_repo/vehicle_reid/automated_training/outputtest_result2.txt", "r") as file:
#                 for line in file:
#                     res_strings = line.split(',')
#                     if res_strings[0] in opt_epochs:
#                         state = str(res_strings[0] + " " + res_strings[1])
#                         value = res_strings[2],res_strings[3].rstrip('\n')
#                         json_data[upp][mid][str(low)][state] = value
                # value = "value"
                # json_matrix[u][m][l][e] = value
                # json_data[upp][mid][str(low)][ep] = value

# print(json_matrix)

# Write the JSON data to a file
# with open("output.json", "w") as json_file:
#     json.dump(json_data, json_file, indent=4)

# # Optional: Print the JSON data
# print(json.dumps(json_data, indent=4))

######################################################################################

# Later, you can fill out the values as needed
# For example, to assign a value to the element at position (i, j, k)
# json_matrix[i][j][k] = value


 
# #os.system('pwd')

# version = list(map(int, torch.__version__.split(".")[:2]))
# torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))


# ######################################################################
# # Options
# # --------
# parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--data_dir', default='../data', type=str, help='path to the dataset root directory')

# parser.add_argument("--train_csv_path", default='../data/(Cityflow)AIC21_Track2_ReID_full/AIC21_Track2_ReID/train_label_split_padded.csv', type=str)

# parser.add_argument("--val_csv_path", default='../data/(Cityflow)AIC21_Track2_ReID_full/AIC21_Track2_ReID/val_label_split_padded.csv', type=str)

# parser.add_argument('--name', default='test_result',
#                     type=str, help='output model name')

# parser.add_argument('--gpu_ids', default='0', type=str,
#                     help='gpu_ids: e.g. 0  0,1,2  0,2')
# parser.add_argument('--tpu_cores', default=-1, type=int,
#                     help="use TPU instead of GPU with the given number of cores (1 recommended if not too many cpus)")
# parser.add_argument('--num_workers', default=3, type=int)
# parser.add_argument('--warm_epoch', default=3, type=int, # te 3 parasti
#                     help='the first K epoch that needs warm up (counted from start_epoch)')
# parser.add_argument('--total_epoch', default=20,
#                     type=int, help='total training epoch')
# parser.add_argument("--save_freq", default=1, type=int,
#                     help="frequency of saving the model in epochs")
# # parser.add_argument("--checkpoint", default="vehicle_reid_repo/vehicle_reid/model/result5/net_20.pth", type=str,
# #                     help="Model checkpoint to load.")
# parser.add_argument("--checkpoint", default="", type=str,
#                     help="Model checkpoint to load.")
# # parser.add_argument("--start_epoch", default=21, type=int,
# #                     help="Epoch to continue training from.")
# parser.add_argument("--start_epoch", default=0, type=int,
#                     help="Epoch to continue training from.")




# parser.add_argument('--fp16', action='store_true',
#                     help='Use mixed precision training. This will occupy less memory in the forward pass, and will speed up training in some architectures (Nvidia A100, V100, etc.)')
# parser.add_argument("--grad_clip_max_norm", type=float, default=50.0,
#                     help="maximum norm of gradient to be clipped to")

# parser.add_argument('--lr', default=0.05,
#                     type=float, help='base learning rate for the head. 0.1 * lr is used for the backbone')
# parser.add_argument('--cosine', action='store_true',
#                     help='use cosine learning rate')
# parser.add_argument('--batchsize', default=32,
#                     type=int, help='batchsize')
# parser.add_argument('--linear_num', default=512, type=int,
#                     help='feature dimension: 512 (default) or 0 (linear=False)')
# parser.add_argument('--stride', default=2, type=int, help='last stride')
# parser.add_argument('--droprate', default=0.5,
#                     type=float, help='drop rate')
# parser.add_argument('--erasing_p', default=0.5, type=float,
#                     help='Random Erasing probability, in [0,1]')
# parser.add_argument('--color_jitter', action='store_true',
#                     help='use color jitter in training')
# parser.add_argument("--label_smoothing", default=0.0, type=float)
# parser.add_argument("--samples_per_class", default=1, type=int,
#                     help="Batch sampling strategy. Batches are sampled from groups of the same class with *this many* elements, if possible. Ordinary random sampling is achieved by setting this to 1.")
                    

# parser.add_argument("--model", default="resnet_ibn",
#                     help="""what model to use, supported values: ['resnet', 'resnet_ibn', densenet', 'swin',
#                     'NAS', 'hr', 'efficientnet'] (default: resnet_ibn)""")
# parser.add_argument("--model_subtype", default="default",
#                     help="Subtype for the model (b0 to b7 for efficientnet)")
# parser.add_argument("--mixstyle", action="store_true",
#                     help="Use MixStyle in training for domain generalization (only for resnet variants yet)")

# parser.add_argument('--arcface', action='store_true',
#                     help='use ArcFace loss')
# parser.add_argument('--circle', action='store_true',
#                     help='use Circle loss')
# parser.add_argument('--cosface', action='store_true',
#                     help='use CosFace loss')
# parser.add_argument('--contrast', action='store_true',
#                     help='use supervised contrastive loss')
# parser.add_argument('--instance', action='store_true',
#                     help='use instance loss')
# parser.add_argument('--ins_gamma', default=32, type=int,
#                     help='gamma for instance loss')
# parser.add_argument('--triplet', default=True, action='store_true',
#                     help='use triplet loss')
# parser.add_argument('--lifted', action='store_true',
#                     help='use lifted loss')
# parser.add_argument('--sphere', action='store_true',
#                     help='use sphere loss')

# parser.add_argument("--debug", action="store_true")
# parser.add_argument("--debug_period", type=int, default=100,
#                     help="Print the loss and grad statistics for *this many* batches at a time.")
# opt = parser.parse_args()

# if opt.label_smoothing > 0.0 and version[0] < 1 or version[1] < 10:
#     warnings.warn(
#         "Label smoothing is supported only from torch 1.10.0, the parameter will be ignored")
    



######################################################################
# Train and evaluate
# ---------------------------


# if version[0] > 1 or (version[0] == 1 and version[1] >= 10):
#     criterion = torch.nn.CrossEntropyLoss(
#         label_smoothing=opt.label_smoothing)
# else:
#     criterion = torch.nn.CrossEntropyLoss()

# model = train_model(
#     model, criterion, start_epoch=opt.start_epoch, num_epochs=opt.total_epoch,
#     num_workers=opt.num_workers
# )


    
# Path to the Python file you want to run
# python_file_path = "vehicle_reid_repo/vehicle_reid/train_modified_for_automation.py"

# # Run the Python file and wait for its execution to finish
# #result = subprocess.run(["python3", python_file_path, "--name=test_result", "--warm_epoch=1", "--lr=0.1", "--cosine", "--fp16", "--stride=1", "--droprate=0.1", "--erasing_p=0.1", "--color_jitter", "--label_smoothing=0.1"], capture_output=False)
# result = subprocess.run(["python3", python_file_path,"--name=test_result","--total_epoch=16", "--fp16", "--cosine"], capture_output=False)
# # Check if the execution was successful
# if result.returncode == 0:
#     print("Script executed successfully!")
#     print(result)
# else:
#     print("An error occurred while executing the script.")
#     print("Error output:", result.stderr.decode())


# # Open the file in read mode
# with open('vehicle_reid_repo/vehicle_reid/automated_training/train_output.txt', 'r') as file:
#     # Create a CSV reader object
#     csv_reader = csv.reader(file)
    
#     # Iterate over each row in the CSV file
#     for row in csv_reader:
#         # Save or process the values in each row
#         # For example, you can print the values
#         print(row[0],row[1],row[2])

