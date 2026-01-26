# -*- coding: utf-8 -*-

'''
!!!
This training script is modified to offer non-garbage validation metrics if the train and val data sets do not have the same vehicle IDs.

Rank and mAP is calculated for different camera Same id's only.

During val we print out Rank-1, mAP and ROC/AUC between positive and negative pairs formed from the val set embeddings.

***26.01*** This revision cleans up the code. PKBatcher is employed to give random batches with class and camera constarints.

Minimal diff summary:

Inside train loop:

*Normalize ff

*Compute ff @ centers.T

*Add CE loss on that

*Remove all pairwise / triplet / margin losses

* Using basic ImageDataset with no cams and basic batcher, not PKBatcher

!!!
'''

from __future__ import print_function, division

import argparse
import time
import os
import sys
import warnings

import numpy as np

import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile
import pandas as pd
import tqdm

from pytorch_metric_learning import losses, miners

# Validation metrics
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

version = list(map(int, torch.__version__.split(".")[:2]))
torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from random_erasing import RandomErasing
from circle_loss import CircleLoss, convert_label_to_similarity
from instance_loss import InstanceLoss
from load_model import load_model_from_opts
from dataset import ImageDataset, ImageDatasetWCam, BatchSampler, CrossCamBatchSampler, PKCrossCamSampler
from custom_losses import OpenWorldBatchLoss, CenterBasedEmbeddingLoss


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--data_dir', default='/home/tomass/tomass/data', type=str, help='path to the dataset root directory')
parser.add_argument('--data_dir', default='/home/tomass/tomass/data', type=str, help='path to the dataset root directory')

# parser.add_argument("--train_csv_path", default='/home/tomass/tomass/data/VeRi/VeRi_train.csv', type=str)
parser.add_argument("--train_csv_path", default='/home/tomass/tomass/data/VeRi_VehicX_labels/train.csv', type=str)

# parser.add_argument("--val_csv_path", default='/home/tomass/tomass/data/VeRi/VeRi_val.csv', type=str)
parser.add_argument("--val_csv_path", default='/home/tomass/tomass/data/VeRi_VehicX_labels/val.csv', type=str)

parser.add_argument('--name', default='student_model_knowl_dist',
                    type=str, help='output model name')

parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--tpu_cores', default=-1, type=int,
                    help="use TPU instead of GPU with the given number of cores (1 recommended if not too many cpus)")
parser.add_argument('--num_workers', default=3, type=int) # Liekas, ka jasamazina worker skaits, ja batch lielaks par 64
parser.add_argument('--warm_epoch', default=0, type=int, # te 3 parasti
                    help='the first K epoch that needs warm up (counted from start_epoch)')
parser.add_argument('--total_epoch', default=20,
                    type=int, help='total training epoch')
parser.add_argument("--save_freq", default=1, type=int, #Originali bija 2
                    help="frequency of saving the model in epochs")
# parser.add_argument("--checkpoint", default="vehicle_reid_repo/vehicle_reid/model/result5/net_20.pth", type=str,
#                     help="Model checkpoint to load.")
parser.add_argument("--checkpoint", default="")
# parser.add_argument("--start_epoch", default=21, type=int,
#                     help="Epoch to continue training from.")
parser.add_argument("--start_epoch", default=0, type=int,
                    help="Epoch to continue training from.")




parser.add_argument('--fp16', action='store_true',
                    help='Use mixed precision training. This will occupy less memory in the forward pass, and will speed up training in some architectures (Nvidia A100, V100, etc.)')
parser.add_argument("--grad_clip_max_norm", type=float, default=50.0,
                    help="maximum norm of gradient to be clipped to")

parser.add_argument('--lr', default=0.05, #0.05 orig
                    type=float, help='base learning rate for the head. 0.1 * lr is used for the backbone')
parser.add_argument('--cosine', action='store_true',
                    help='use cosine learning rate')
parser.add_argument('--batchsize', default=32,
                    type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, #default=512
                    help='feature dimension: 512 (default) or 0 (linear=False)')
parser.add_argument('--stride', default=1, type=int, help='last stride') #default=2
parser.add_argument('--droprate', default=0.5,
                    type=float, help='drop rate')
parser.add_argument('--erasing_p', default=0.5, type=float,
                    help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter',default=True, action='store_true', # parasti nav
                    help='use color jitter in training')
parser.add_argument("--label_smoothing", default=0.0, type=float)
parser.add_argument("--samples_per_class", default=4, type=int,
                    help="Batch sampling strategy. Batches are sampled from groups of the same class with *this many* elements, if possible. Ordinary random sampling is achieved by setting this to 1.")
parser.add_argument("--samples_per_camera", default=2, type=int,
                    help="Batch sampling strategy. Batches are sampled in a way that respects the minimum number of different cameras that an ID must have in batch.")
parser.add_argument("--batches_per_epoch_num", default=1500, type=int,
                    help="Number of baches per epoch to run if batches are constructed by PK_batcher and are randomised.")

parser.add_argument("--model", default="resnet_ibn",
                    help="""what model to use, supported values: ['resnet', 'resnet_ibn', densenet', 'swin',
                    'NAS', 'hr', 'efficientnet'] (default: resnet_ibn)""")
parser.add_argument("--model_subtype", default="default",
                    help="Subtype for the model (b0 to b7 for efficientnet)")
parser.add_argument("--mixstyle", action="store_true",
                    help="Use MixStyle in training for domain generalization (only for resnet variants yet)")

parser.add_argument('--arcface', action='store_true',
                    help='use ArcFace loss')
parser.add_argument('--circle', action='store_true',
                    help='use Circle loss')
parser.add_argument('--cosface', action='store_true',
                    help='use CosFace loss')
parser.add_argument('--contrast', action='store_true',
                    help='use supervised contrastive loss')
parser.add_argument('--instance', action='store_true',
                    help='use instance loss')
parser.add_argument('--ins_gamma', default=32, type=int,
                    help='gamma for instance loss')
parser.add_argument('--triplet', action='store_true',
                    help='use triplet loss')
parser.add_argument('--lifted', action='store_true',
                    help='use lifted loss')
parser.add_argument('--sphere', action='store_true',
                    help='use sphere loss')
parser.add_argument('--center_based', default=True, action='store_true',
                    help='use center based embedding loss for knowledge distillation')

parser.add_argument("--debug", action="store_true")
parser.add_argument("--debug_period", type=int, default=100,
                    help="Print the loss and grad statistics for *this many* batches at a time.")
opt = parser.parse_args()


if opt.label_smoothing > 0.0 and version[0] < 1 or version[1] < 10:
    warnings.warn(
        "Label smoothing is supported only from torch 1.10.0, the parameter will be ignored")



######################################################################
# Configure devices
# ---------
#

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name


gpu_ids = []
if opt.gpu_ids:
    str_ids = opt.gpu_ids.split(',')
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

use_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
if not use_gpu:
    print("Running on CPU ...")
else:
    print("Running on cuda:{}".format(gpu_ids[0]))
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#

cams_per_id = opt.samples_per_camera
num_batches = opt.batches_per_epoch_num

h, w = 224, 224
interpolation = 3 if torchvision_version[0] == 0 and torchvision_version[1] < 13 else \
    transforms.InterpolationMode.BICUBIC

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((h, w), interpolation=interpolation),
    transforms.Pad(10),
    transforms.RandomCrop((h, w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(h, w), interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + \
        [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print("Train transforms:", transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {}
train_df = pd.read_csv(opt.train_csv_path)
val_df = pd.read_csv(opt.val_csv_path)
all_ids = list(set(train_df["id"]).union(set(val_df["id"])))
image_datasets["train"] = ImageDataset(
    opt.data_dir, train_df, "id", classes=all_ids, transform=data_transforms["train"])
# no-cam IDs included
# image_datasets["val"] = ImageDataset(
#     opt.data_dir, val_df, "id", classes=all_ids, transform=data_transforms["val"])
# cam IDs included
image_datasets["val"] = ImageDatasetWCam(
    opt.data_dir, val_df, "id", classes=all_ids, transform=data_transforms["val"])


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
opt.nclasses = len(class_names)
print("Number of classes in total: {}".format(opt.nclasses))

######################################################################
# Some Utilities for training
#


class DebugInfo:
    def __init__(self, name, print_period):
        self.history = []
        self.name = name
        self.print_period = print_period

    def step(self, value):
        self.history.append(value)
        if len(self.history) >= self.print_period:
            print("\n{}:".format(self.name))
            print(pd.Series(self.history).describe())
            self.history = []


######################################################################
# Training the model
# ------------------
# loss history
y_loss = {}
y_loss['train'] = []
y_loss['val'] = []

# error history, error = 1 - accuracy
y_err = {}
y_err['train'] = []
y_err['val'] = []


def fliplr(img):
    """flip a batch of images horizontally"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -
                           1).long().to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip



def train_model(model, criterion, teacher, start_epoch=0, num_epochs=25, num_workers=2):
    
    since = time.time()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    teacher = teacher.to(device)

    if fp16:
        scaler = amp.GradScaler()
        autocast = amp.autocast()

    # create optimizer and scheduler
    optim_name = optim.SGD
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(
        p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()
    optimizer = optim_name([
        {'params': base_params, 'initial_lr': 0.1 * opt.lr, ### Origininali 0.1 * 
         'lr': 0.1 * opt.lr},
        {'params': classifier_params, 'initial_lr': opt.lr,
         'lr': opt.lr},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)



    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)
    if opt.cosine:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, opt.total_epoch, eta_min=0.01 * opt.lr)

    for _ in range(start_epoch):
        scheduler.step()

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(
        dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch
    
    ###Debug


    # initialize losses
    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(
            num_classes=opt.nclasses, embedding_size=512).to(device)
    if opt.cosface:
        criterion_cosface = losses.CosFaceLoss(
            num_classes=opt.nclasses, embedding_size=512).to(device)
    if opt.circle:
        # gamma = 64 may lead to a better result.
        criterion_circle = CircleLoss(m=0.25, gamma=32).to(device)
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3).to(device)
    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(
            neg_margin=1, pos_margin=0).to(device)
    if opt.contrast:
        criterion_contrast = losses.ContrastiveLoss(
            pos_margin=0, neg_margin=0.4).to(device)
    if opt.instance:
        criterion_instance = InstanceLoss(gamma=opt.ins_gamma).to(device)
    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(
            num_classes=opt.nclasses, embedding_size=512, margin=4).to(device)
    if opt.center_based:
        criterion_center_based = CenterBasedEmbeddingLoss(scale=32).to(device)


    train_sampler = BatchSampler(
        image_datasets["train"], opt.batchsize, opt.samples_per_class)

    dataloaders = {
        "val": torch.utils.data.DataLoader(image_datasets["val"],
                                            batch_size=opt.batchsize,
                                            num_workers=num_workers,
                                            pin_memory=use_gpu),
        
        "train": torch.utils.data.DataLoader(image_datasets["train"],
                                                batch_sampler=train_sampler,
                                                num_workers=num_workers,
                                                pin_memory=use_gpu)
    }

    ### DEBUG for Cam usage >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # loader = dataloaders['train']
    # batch = next(iter(loader))

    # _, labels, cams = batch

    # from collections import defaultdict
    # id2cams = defaultdict(set)

    # for l, c in zip(labels.tolist(), cams.tolist()):
    #     id2cams[l].add(c)

    # for l, cams_used in id2cams.items():
    #     assert len(cams_used) >= cams_per_id

    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    ####################################################################
    # Computing class centers for knowledge distillation
    # ------------------------------------------------------

    num_classes = opt.nclasses
    feat_dim = opt.linear_num if opt.linear_num > 0 else 512

    centers = torch.zeros(num_classes, feat_dim) ## Okey te gan es nevaru saprast vai tika pienemts ka visi ID ir 0 lidz N-1, nevis tie kas ir no csv
    counts = torch.zeros(num_classes)

    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(dataloaders['train'], desc="Computing class centers"):

            assert labels.min() >= 0
            assert labels.max() < num_classes

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = teacher(imgs)

            if isinstance(outputs, (tuple, list)):
                feats = outputs[-1]
            else:
                feats = outputs

            feats = F.normalize(feats, dim=1)

            for feat, label in zip(feats, labels):
                centers[label] += feat.cpu()
                counts[label] += 1

        centers = centers / counts.unsqueeze(1)
        centers = F.normalize(centers, dim=1)

        centers = centers.to(device)
        centers.requires_grad = False

        assert centers.shape == (num_classes, feat_dim)
        assert not centers.requires_grad

        ### DEBUG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        sim = centers @ centers.T
        print("Class centers similarity matrix statistics:")
        print(sim.mean().item(), sim.diag().mean().item()) # diag should be 1.0, mean should be low ~ 0.0
        ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    mean_diff = 0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        
        ###################### TRAIN PHASE ############################
        loader = tqdm.tqdm(dataloaders['train'])

        model.train(True)

        running_loss = torch.zeros(1).to(device)
        running_corrects = torch.zeros(1).to(device)

        if opt.debug:
            loss_debug = DebugInfo("loss", opt.debug_period)
            grad_debug = DebugInfo("grad", opt.debug_period)

        for batch_idx, data in enumerate(loader):
            if batch_idx >= num_batches:
                break
            inputs, labels = data
            now_batch_size = inputs.shape[0]

            if use_gpu:
                inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass

            if fp16:
                autocast.__enter__()

            outputs = model(inputs)

            if return_feature:
                logits, ff = outputs

                #loss = 0.0
                loss = criterion(logits, labels)
                
                ff = F.normalize(ff, dim=1)

                
                # DEBUG STATS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                with torch.no_grad():
                    sim = ff @ ff.t()          # cosine similarity
                    labels_eq = labels[:, None] == labels[None, :]

                    pos = sim[labels_eq].view(-1)
                    neg = sim[~labels_eq].view(-1)
                    diff = pos.mean() - neg.mean()

                    mean_diff += diff

                loader.set_postfix({
                    "pos": f"{pos.mean():.3f}",
                    "neg": f"{neg.mean():.3f}",
                    "diff": f"{diff:.3f}"
                })
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                
                
                #loss += criterion_open(ff,labels)

                if opt.arcface:
                    loss += criterion_arcface(ff, labels) / now_batch_size
                if opt.cosface:
                    loss += criterion_cosface(ff, labels) / now_batch_size
                if opt.circle:
                    loss += criterion_circle(
                        *convert_label_to_similarity(ff, labels)) / now_batch_size
                if opt.triplet:
                    hard_pairs = miner(ff, labels)
                    triplet_loss = criterion_triplet(ff, labels, hard_pairs)
                    loss += 0.5 * triplet_loss
                if opt.lifted:
                    loss += criterion_lifted(ff, labels)  # /now_batch_size
                if opt.contrast:
                    # / now_batch_size
                    loss += 0.5 * criterion_contrast(ff, labels)
                if opt.instance:
                    loss += criterion_instance(ff, labels) / now_batch_size
                if opt.sphere:
                    loss += criterion_sphere(ff, labels) / now_batch_size
                if opt.center_based:
                    loss += criterion_center_based(ff, labels, centers)

                
            else:
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

            if opt.debug:
                loss_debug.step(loss.item())

            # adjust loss by warmup learning rate if applicable
            if epoch < opt.warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss = loss * warm_up

            # backward + optimize only if in training phase
            if fp16:
                autocast.__exit__(None, None, None)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            # perform gradient clipping to prevent divergence
            old_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), opt.grad_clip_max_norm)

            if opt.debug:
                grad_debug.step(old_norm.item())

            #optimizer step
            if fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            #DEBUG for initialization
            # if epoch == 0:
            #     print("Grad norm:", old_norm.item())

            running_loss += loss.item() * now_batch_size

            if not return_feature:
                running_corrects += float(torch.sum(preds == labels.data))

        mean_diff = mean_diff / num_batches
        epoch_loss = running_loss.cpu() / dataset_sizes['train']
        if return_feature:
            print('{} Loss: {:.4f} Mean_diff: {:.4f}'.format(
                'train', epoch_loss.item(), mean_diff))
        else:
            epoch_acc = running_corrects.cpu() / dataset_sizes['train']
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss.item(), epoch_acc.item()))

        scheduler.step()


        
        with open("vehicle_reid_repo2/vehicle_reid/automated_training/"+ opt.name +".txt", "a") as file:
            # Write some lines to the file

            if return_feature:
                epoch_acc = 0.0
                file.write('{},{},{:.4f},{:.4f}\n'.format(
                    epoch,'train', epoch_loss.item(), mean_diff))
            else:
                file.write('{},{},{:.4f},{:.4f}\n'.format(
                            epoch, 'train', epoch_loss.item(), epoch_acc.item()))


        if epoch == num_epochs - 1 or (epoch % (opt.save_freq) == (opt.save_freq - 1)):
            save_network(model, epoch)
    #        draw_curve(epoch)
        
        #Šito aizkomentet pectam
        # print('{},{},{:.4f},{:.4f}\n'.format(
        #     epoch, 'train', epoch_loss.item(), epoch_acc.item()))



        time_elapsed = time.time() - since
        print('Epoch complete at {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

        ##################### VAL PHASE ############################

        metrics = evaluate_reid(model, dataloaders["val"], device)

        print("Open-world Val:")
        print(f"  open_world_acc: {metrics['open_world_acc']:.4f}")
        print(f"  Mean margin: {metrics['mean_margin']:.4f}")
        print(f"  10th percentile margin: {metrics['p10_margin']:.4f}")
        print(f"  Number of valid queries: {metrics['num_valid_queries']}")

        with open("vehicle_reid_repo2/vehicle_reid/automated_training/"+ opt.name +".txt", "a") as file:
            # Write some lines to the file
            file.write('{},{},{:.4f},{:.4f},{:.4f}\n'.format(
            epoch, 'val', metrics['open_world_acc'], metrics['mean_margin'], metrics['p10_margin']))


        time_elapsed = time.time() - since
        print('Epoch complete at {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


######################################################################
# Open-world ReID evaluation metric
# ------------------------------------------------------

@torch.no_grad()
def open_world_proxy_crosscam_gpu(
    feats,  # torch.Tensor [N, D], normalized, CUDA
    ids,    # torch.Tensor [N], CPU or CUDA
    cams,   # torch.Tensor [N], CPU or CUDA
    num_bg=None,
    seed=0,
):
    
    """
    Open-world ReID proxy using all images as queries, all cross-camera positives.
    
    Args:
        feats: np.array of shape [N, D], normalized feature vectors
        ids: np.array of shape [N], integer IDs
        cams: np.array of shape [N], camera indices
        num_bg: int or None, number of negatives to sample per query
        seed: random seed
    
    Returns:
        dict with:
            open_world_acc: fraction of positive>max_negative
            mean_margin: mean(sim_pos - max_neg)
            p10_margin: 10th percentile margin
            num_valid_queries: total number of valid query-positive pairs


    AKA picks a random query and a positive from a different camera, then
    samples num_bg negatives for this particular embedding from different IDs. Computes whether this one random pair positive
    similarity is larger than the maximum negative similarity, and the margin
    between them.
    """
    
    start_time = time.time()
 

    device = feats.device
    torch.manual_seed(seed)

    N, D = feats.shape
    correct = 0
    margins = []

    ids = ids.to(device)
    cams = cams.to(device)

    all_idx = torch.randperm(N, device=device)

    for q in all_idx:
        q_id = ids[q]
        q_cam = cams[q]
        qf = feats[q]                      # [D]

        # cross-camera positives
        pos_mask = (ids == q_id) & (cams != q_cam)
        pos_idx = torch.where(pos_mask)[0]
        if pos_idx.numel() == 0:
            continue

        # negatives
        neg_idx_all = torch.where(ids != q_id)[0]
        if num_bg is not None and neg_idx_all.numel() > num_bg:
            perm = torch.randperm(neg_idx_all.numel(), device=device)[:num_bg]
            neg_idx = neg_idx_all[perm]
        else:
            neg_idx = neg_idx_all

        bgf = feats[neg_idx]               # [num_bg, D]
        sim_neg_max = torch.max(bgf @ qf)  # scalar

        # all positives for this query
        pf = feats[pos_idx]                # [P, D]
        sim_pos = pf @ qf                  # [P]
        margin = sim_pos - sim_neg_max     # [P]

        margins.append(margin)
        correct += torch.sum(margin > 0).item()

    margins = torch.cat(margins)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time} seconds")

    return {
        "open_world_acc": correct / margins.numel(),
        "mean_margin": margins.mean().item(),
        "p10_margin": torch.quantile(margins, 0.10).item(),
        "num_valid_queries": margins.numel(),
    }

@torch.no_grad()
def evaluate_reid(model, dataloader, device):
    model.eval()

    all_feats = []
    all_ids = []
    all_cams = []

    for inputs, labels, cams in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        # Handle all common ReID model output formats
        if isinstance(outputs, (tuple, list)):
            # Common cases:
            # (logits, feat)
            # [logits, feat]
            # [feat] or [feat1, feat2, ...]
            feats = outputs[-1]
        else:
            feats = outputs

        assert isinstance(feats, torch.Tensor), type(feats)
        feats = torch.nn.functional.normalize(feats, dim=1)

        all_feats.append(feats)
        all_ids.append(labels)
        all_cams.append(cams)

    feats = torch.cat(all_feats, dim=0).to(device)   # KEEP ON GPU
    ids = torch.cat(all_ids, dim=0)
    cams = torch.cat(all_cams, dim=0)

    results = open_world_proxy_crosscam_gpu(feats, ids, cams)

    return results


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join(SCRIPT_DIR, "model", name, 'train.jpg'))

######################################################################
# Save model
# ---------------------------


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(SCRIPT_DIR, "model", name, save_filename)
    device = next(iter(network.parameters())).device
    torch.save(network.cpu().state_dict(), save_path)
    network.to(device)

######################################################################
# Load Teacher Model
# ---------------------------

teach_dir = os.path.join(SCRIPT_DIR, "model", "base_CE_teacher")
teach_opts_file = "%s/opts.yaml" % teach_dir
teach_epoch = 15
teach_checkpoint = os.path.join(teach_dir, f"net_{teach_epoch}.pth")  # or net_best.pth

return_teach_feature = True

teacher_model = load_model_from_opts(teach_opts_file,
                             ckpt=teach_checkpoint,
                             return_feature=return_teach_feature)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False


######################################################################
# Save opts and load student model
# ---------------------------

dir_name = os.path.join(SCRIPT_DIR, "model", name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run
copyfile(os.path.join(SCRIPT_DIR, 'train.py'),
         os.path.join(dir_name, "train.py"))
copyfile(os.path.join(SCRIPT_DIR, "model.py"),
         os.path.join(dir_name, "model.py"))

# save opts
opts_file = "%s/opts.yaml" % dir_name
with open(opts_file, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere

student_model = load_model_from_opts(opts_file,
                             ckpt=opt.checkpoint if opt.checkpoint else None,
                             return_feature=return_feature)
# model is on CPU at this point, we send it to the device in the training function
student_model.train()


######################################################################
# Train and evaluate
# ---------------------------

if version[0] > 1 or (version[0] == 1 and version[1] >= 10):
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=opt.label_smoothing)
else:
    criterion = torch.nn.CrossEntropyLoss()

model = train_model(
    student_model, criterion, start_epoch=opt.start_epoch, num_epochs=opt.total_epoch,
    num_workers=opt.num_workers, teacher=teacher_model
)
