import numpy as np
import torch
import tqdm

import torchvision
from torchvision import transforms

######################################################################
# Load Data
# ---------
#

torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))


h, w = 224, 224
interpolation = 3 if torchvision_version[0] == 0 and torchvision_version[1] < 13 else \
    transforms.InterpolationMode.BICUBIC

transform_val_list = [
    transforms.Resize(size=(h, w), interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'val': transforms.Compose(transform_val_list),
}

def open_world_proxy_crosscam(
    feats,
    ids,
    cams,
    num_bg=8192,
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


    #random sample generator
    rng = np.random.default_rng(seed)
    N = len(ids)
    correct = 0
    margins = []

    all_idx = np.arange(N)
    rng.shuffle(all_idx)

    for q in all_idx:
        q_id = ids[q]
        q_cam = cams[q]
        qf = feats[q]

        # cross-camera positives
        pos_idx = np.where((ids == q_id) & (cams != q_cam))[0]
        if len(pos_idx) == 0:
            continue

        # background negatives: all other IDs
        bg_idx_all = np.where(ids != q_id)[0]
        if num_bg is not None and len(bg_idx_all) > num_bg:
            bg_idx = rng.choice(bg_idx_all, size=num_bg, replace=False)
        else:
            bg_idx = bg_idx_all

        bgf = feats[bg_idx]

        # compute sim for all positives
        for p in pos_idx:
            pf = feats[p]
            sim_pos = float(np.dot(qf, pf))
            sim_neg_max = float(np.max(bgf @ qf))
            margin = sim_pos - sim_neg_max
            margins.append(margin)
            if margin > 0:
                correct += 1

    margins = np.array(margins)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed Time: {elapsed_time} seconds")
    return {
        "open_world_acc": correct / len(margins) if len(margins) else 0.0,
        "mean_margin": float(np.mean(margins)) if len(margins) else 0.0,
        "p10_margin": float(np.percentile(margins, 10)) if len(margins) else 0.0,
        "num_valid_queries": len(margins),
    }

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

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    import torch
    import argparse

    import time

    CLIP = True

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPT_DIR)

    if CLIP:
        import clip
        import counting_workspace.misc.feature_extract_CLIP as fExtract
        from vehicle_reid_repo2.vehicle_reid.load_model import load_CLIP_head_from_opts
        from vehicle_reid_repo2.vehicle_reid.dataset import ImageDatasetWCamCLIP
    else:
        from vehicle_reid_repo2.vehicle_reid.load_model import load_model_from_opts
        from vehicle_reid_repo2.vehicle_reid.dataset import ImageDatasetWCam


    data_dir = "/home/tomass/tomass/data/VeRi"
    csv_path = "/home/tomass/tomass/data/VeRi/val.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu = torch.cuda.is_available()

    image_datasets = {}
    val_df = pd.read_csv(csv_path)
    all_ids = list(set(val_df["id"]))

    if CLIP:

        clip_model_name = "ViT-B/32"

        clip_model, preprocess = clip.load(clip_model_name, device=device)
        clip_model = clip_model.float()

        model = load_CLIP_head_from_opts(
            "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_64b_4pc/opts.yaml",
            clip_visual=clip_model.visual,
            ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_64b_4pc/net_19.pth",
            remove_classifier=True,
        )
        model.eval()
        model.to(device)

        image_datasets["val"] = ImageDatasetWCamCLIP(
            data_dir, val_df, "id", classes=all_ids, transform=preprocess, device=device)
        
    else:
        model = load_model_from_opts(
            "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/student_model_knowl_dist/opts.yaml",
            ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/student_model_knowl_dist/net_0.pth",
            remove_classifier=True,
        )
        model.eval()
        model.to(device)

        image_datasets["val"] = ImageDatasetWCam(
            data_dir, val_df, "id", classes=all_ids, transform=data_transforms["val"])

    dataloaders = {
        "val": torch.utils.data.DataLoader(image_datasets["val"],
                                            batch_size=32,
                                            num_workers=3,
                                            pin_memory=use_gpu),
        
        "train": None,
    }

    results = evaluate_reid(model, dataloaders["val"], device)

    print("Open-world ReID proxy (cross-camera positives only) results:")
    print(f"  open_world_acc: {results['open_world_acc']:.4f}")
    print(f"  Mean margin: {results['mean_margin']:.4f}")
    print(f"  10th percentile margin: {results['p10_margin']:.4f}")
    print(f"  Number of valid queries: {results['num_valid_queries']}")
