import numpy as np
import torch
import tqdm

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

        all_feats.append(feats.cpu())
        all_ids.append(labels.cpu())
        all_cams.append(cams.cpu())

    feats = torch.cat(all_feats, dim=0).numpy()
    ids = torch.cat(all_ids, dim=0).numpy()
    cams = torch.cat(all_cams, dim=0).numpy()

    # ---- ReID metrics ----
    results = open_world_proxy_crosscam(feats, ids, cams)

    return results

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    import torch
    import argparse

    import time

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(SCRIPT_DIR)

    import clip
    import counting_workspace.misc.feature_extract_CLIP as fExtract
    from vehicle_reid_repo2.vehicle_reid.load_model import load_CLIP_head_from_opts
    from vehicle_reid_repo2.vehicle_reid.dataset import ImageDatasetWCamCLIP


    data_dir = "/home/tomass/tomass/data/VeRi"
    csv_path = "/home/tomass/tomass/data/VeRi/val.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu = torch.cuda.is_available()

    clip_model_name = "ViT-B/32"

    clip_model, preprocess = clip.load(clip_model_name, device=device)
    clip_model = clip_model.float()

    image_datasets = {}
    val_df = pd.read_csv(csv_path)
    all_ids = list(set(val_df["id"]))

    image_datasets["val"] = ImageDatasetWCamCLIP(
        data_dir, val_df, "id", classes=all_ids, transform=preprocess, device=device)

    dataloaders = {
        "val": torch.utils.data.DataLoader(image_datasets["val"],
                                            batch_size=32,
                                            num_workers=3,
                                            pin_memory=use_gpu),
        
        "train": None
    }
    
    model = load_CLIP_head_from_opts(
        "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/opts.yaml",
        clip_visual=clip_model.visual,
        ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/net_10.pth",
        remove_classifier=True,
    )
    model.eval()
    model.to(device)

    results = evaluate_reid(model, dataloaders["val"], device)

    print("Open-world ReID proxy (cross-camera positives only) results:")
    print(f"  open_world_acc: {results['open_world_acc']:.4f}")
    print(f"  Mean margin: {results['mean_margin']:.4f}")
    print(f"  10th percentile margin: {results['p10_margin']:.4f}")
    print(f"  Number of valid queries: {results['num_valid_queries']}")
