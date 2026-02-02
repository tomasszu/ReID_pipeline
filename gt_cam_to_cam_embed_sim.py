"""
This code is used to compute the similarity of embeddings between different views of the same vehicle
from different camera ( or one fisheye camera) perspectives.

Originally set up for the following structure:
- Dates video taken
    - Camera (here using only one fisheye camera)
        - Perspective (Left, Right, Center)

Each perspective or the bottom unit is supposed to contain an annotation file

"""

import os
import sys
import pandas as pd
import numpy as np

import torchvision
import torch
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image

import tqdm
from itertools import combinations
from collections import defaultdict

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


def load_csv(main_dir, date_folders, camera_folders, perspective_folders):
    
    #disctionary to hold a csv dataframe for each perspective
    dfs = []

    
    for date_folder in date_folders:
        for camera_folder in camera_folders:
            for perspective_folder in perspective_folders:
                annotation_file = f"{main_dir}/{date_folder}/{camera_folder}/{perspective_folder}/annotations.csv"

                df = pd.read_csv(annotation_file)

                df["perspective"] = perspective_folder
                df["camera"] = camera_folder
                df["date"] = date_folder
                dfs.append(df)
                

    
    print("dataframe loaded successfully.")
    return pd.concat(dfs, ignore_index=True)

def load_embeddings(df, model):
    buckets = {}  # perspective -> {"embeddings": [], "vehicle_ids": []}

    print("Loading embeddings...")
    for _, row in tqdm.tqdm(dataframe.iterrows()):
        perspective = row["perspective"]

        if perspective not in buckets:
            buckets[perspective] = {
                "embeddings": [],
                "vehicle_ids": []
            }

        full_img_path = os.path.join(
            main_dir,
            row["date"],
            row["camera"],
            row["filename"]
        )

        image = Image.open(full_img_path).convert("RGB")

        if CLIP:
            image = preprocess(image)
        else:
            image = data_transforms['val'](image)

        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(image) # [1, D]

        emb = emb.squeeze(0)  # [D]

        buckets[perspective]["embeddings"].append(emb)
        buckets[perspective]["vehicle_ids"].append(row["track_id"])

    # Finalize: stack on CUDA
    print("Stacking on CUDA...")
    for p in tqdm.tqdm(buckets):
        buckets[p]["embeddings"] = torch.stack(
            buckets[p]["embeddings"], dim=0
        )  # [N, D] on CUDA

        buckets[p]["vehicle_ids"] = torch.tensor(
            buckets[p]["vehicle_ids"], device=device
        )   # [N] on CUDA

        # print(f"Perspective {p} DEBUG:")
        # print(f"Loaded {buckets[p]['embeddings'].shape[0]} embeddings for perspective {p}")
        # print(f"Vehicle IDs tensor shape: {buckets[p]['vehicle_ids'].shape}")
        # print(f"Sample vehicle IDs: {buckets[p]['vehicle_ids'][:100].cpu().numpy()}")

    return buckets   


def cross_cam_similarity(bucket_a, bucket_b):
    """
    bucket_*:
        embeddings: [N, D] CUDA
        vehicle_ids: [N]
    """

    A = F.normalize(bucket_a["embeddings"], dim=1)
    B = F.normalize(bucket_b["embeddings"], dim=1)

    sim = A @ B.T  # [Na, Nb]

    id_match = (
        bucket_a["vehicle_ids"][:, None]
        == bucket_b["vehicle_ids"][None, :]
    )

    pos = sim[id_match]      # same vehicle, different camera
    neg = sim[~id_match]     # background

    return pos, neg

def compute_stats(embeddings_buckets):
    all_stats = {}
    id_accumulator = defaultdict(lambda: {
        "pos": [],
        "neg": []
    })

    print("Computing cross camera similarities...")

    for cam_a, cam_b in combinations(embeddings_buckets.keys(), 2):
        A = embeddings_buckets[cam_a]
        B = embeddings_buckets[cam_b]

        A_emb = F.normalize(A["embeddings"], dim=1)
        B_emb = F.normalize(B["embeddings"], dim=1)

        sim = A_emb @ B_emb.T  # [Na, Nb]

        ids_A = A["vehicle_ids"]
        ids_B = B["vehicle_ids"]

        id_match = ids_A[:, None] == ids_B[None, :]

        pos = sim[id_match]
        neg = sim[~id_match]

        all_stats[(cam_a, cam_b)] = {
            "pos": pos,
            "neg": neg
        }

        # ---- ID-level accumulation ----
        for i in range(sim.size(0)):
            vid = int(ids_A[i].item())

            row_sim = sim[i]
            row_match = id_match[i]

            if row_match.any():
                id_accumulator[vid]["pos"].append(
                    row_sim[row_match]
                )

            id_accumulator[vid]["neg"].append(
                row_sim[~row_match]
            )

    # ---- Finalize per-ID stats ----
    id_stats = {}

    for vid, vals in id_accumulator.items():
        pos = torch.cat(vals["pos"]) if len(vals["pos"]) else None
        neg = torch.cat(vals["neg"]) if len(vals["neg"]) else None

        if pos is None or neg is None:
            continue

        pos_mean = pos.mean().item()
        neg_mean = neg.mean().item()

        id_stats[vid] = {
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "margin": pos_mean - neg_mean,
            "pos_count": pos.numel(),
            "neg_count": neg.numel()
        }

    return all_stats, id_stats

def summarize(sims):
    return {
        "mean": sims.mean().item(),
        "std": sims.std().item(),
        "p10": sims.quantile(0.10).item(),
        "SD-": sims.quantile(0.32).item(),
        "SD+": sims.quantile(0.68).item(),
        "p90": sims.quantile(0.90).item(),
        "count": sims.numel()
    }

def average_precision(sorted_matches):
    """
    sorted_matches: Bool tensor [K] sorted by similarity (True = correct ID)
    """
    if not sorted_matches.any():
        return 0.0

    correct = sorted_matches.float()
    precision_at_k = correct.cumsum(0) / torch.arange(
        1, len(correct) + 1, device=correct.device
    )

    ap = (precision_at_k * correct).sum() / correct.sum()
    return ap.item()


def evaluate_rank1_map(embeddings_buckets):
    rank1_hits = []
    aps = []

    cameras = list(embeddings_buckets.keys())

    for cam_q, cam_g in combinations(cameras, 2):
        Q = embeddings_buckets[cam_q]
        G = embeddings_buckets[cam_g]

        q_emb = F.normalize(Q["embeddings"], dim=1)
        g_emb = F.normalize(G["embeddings"], dim=1)

        sim = q_emb @ g_emb.T  # [Nq, Ng]

        q_ids = Q["vehicle_ids"]
        g_ids = G["vehicle_ids"]

        for i in range(sim.size(0)):
            scores = sim[i]                     # [Ng]
            matches = (g_ids == q_ids[i])       # [Ng]

            if not matches.any():
                continue  # no GT in gallery → skip (standard)

            order = torch.argsort(scores, descending=True)
            sorted_matches = matches[order]

            # Rank-1
            rank1_hits.append(sorted_matches[0].item())

            # AP
            aps.append(average_precision(sorted_matches))

    rank1 = sum(rank1_hits) / max(len(rank1_hits), 1)
    mAP = sum(aps) / max(len(aps), 1)

    return {
        "Rank-1": rank1,
        "mAP": mAP,
        "num_queries": len(rank1_hits)
    }


if __name__ == "__main__":
    
    main_dir = "/home/tomass/tomass/Cam_record"
    date_folders = ["04.09.25_2","04.09.25_3","12.01.26","13.01.26","14.01.26"]
    camera_folders = ["perspective_views_fisheye_record"]
    perspective_folders = ["center", "left", "right"]

    CLIP = False

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu = torch.cuda.is_available()

    no_class_block = True

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
        
    else:
        model = load_model_from_opts(
            "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CE_triplet_no_batchnorm/opts.yaml",
            ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CE_triplet_no_batchnorm/net_0.pth",
            remove_classifier=True,
            return_pre_bn=True)
        model.eval()
        model.to(device)


    dataframe = load_csv(main_dir, date_folders, camera_folders, perspective_folders)

    embeddings_buckets = load_embeddings(dataframe, model)

    # First we calculate (cross-perspective) Rank-1 and mAP
    results = evaluate_rank1_map(embeddings_buckets)

    print(results)

    stats, id_stats = compute_stats(embeddings_buckets)

    hard_ids = sorted(
        id_stats.items(),
        key=lambda x: x[1]["margin"]
    )[:10]

    print("\n ======== HARDEST IDS ======== \n")
    for vid, s in hard_ids:
        print(f"Vehicle ID {vid}:")
        print(f"  Pos mean: {s['pos_mean']:.4f}")
        print(f"  Neg mean: {s['neg_mean']:.4f}")
        print(f"  Margin: {s['margin']:.4f}")
        print(f"  Pos count: {s['pos_count']}")
        print(f"  Neg count: {s['neg_count']}")
        print("")

    diff_stats = {}
    print("\n ======== PER CAMERA PAIR STATS ======== \n")
    for cams, s in stats.items():
        print(f"\n -------------------{cams}-------------------- \n")
        pos = summarize(s["pos"])
        neg = summarize(s["neg"])
        print("POS:", pos)
        print("NEG:", neg)
        diff_stats[cams] = {
                "p10_diff": pos["p10"] - neg["p90"],
                "SD_diff": pos["SD-"] - neg["SD+"],
                "mean_diff": pos["mean"] - neg["mean"],
            }
    
    print("\n ======== DIFFERENCE STATS ======== \n")
    p_10_diffs = []
    p_25_diffs = []
    mean_diffs = []
    for cams, ds in diff_stats.items():
        p_10_diffs.append(ds["p10_diff"])
        p_25_diffs.append(ds["SD_diff"])
        mean_diffs.append(ds["mean_diff"])
    
    print("Average P10 difference: ", np.mean(p_10_diffs))
    print("Average SD border difference: ", np.mean(p_25_diffs))
    print("Average Mean difference: ", np.mean(mean_diffs))




    #print("Embeddings dictionary :", embeddings_buckets)

