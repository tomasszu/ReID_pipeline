import torch
from collections import defaultdict
from itertools import combinations

def inspect_batches(dataset, batch_sampler, max_batches=5):
    """
    Inspect batches for ReID / metric learning.

    Args:
        dataset: your Dataset object (returns image, label, cam_id)
        batch_sampler: your BatchSampler object
        max_batches: number of batches to inspect
    """

    batch_iter = iter(batch_sampler)  # Use the iterator
    for batch_idx in range(max_batches):
        try:
            batch_indices = next(batch_iter)
        except StopIteration:
            break

        # Gather labels and cam_ids
        labels, cam_ids = [], []
        for idx in batch_indices:
            _, label, cam_id = dataset[idx]
            labels.append(label)
            cam_ids.append(cam_id)

        labels = torch.tensor(labels)
        # cam_ids are optional; convert to tensor if numeric, else keep as list
        cam_ids_tensor = None
        try:
            cam_ids_tensor = torch.tensor(cam_ids)
        except:
            cam_ids_tensor = cam_ids

        # Count IDs
        unique_ids = set(labels.tolist())
        n_ids = len(unique_ids)
        n_imgs = len(labels)

        # Count positive pairs per ID
        pos_pairs = 0
        neg_pairs = 0
        for id_val in unique_ids:
            id_indices = (labels == id_val).nonzero(as_tuple=True)[0]
            pos_pairs += len(list(combinations(id_indices.tolist(), 2)))
            neg_pairs += len(id_indices) * (n_imgs - len(id_indices))

        print(f"Batch {batch_idx}:")
        print(f"  Total images: {n_imgs}")
        print(f"  Unique IDs: {n_ids}")
        print(f"  Positive pairs: {pos_pairs}")
        print(f"  Negative pairs: {neg_pairs}")
        if cam_ids_tensor is not None and isinstance(cam_ids_tensor, torch.Tensor):
            # Count cross-camera positives if numeric
            cross_cam_pos = 0
            for id_val in unique_ids:
                id_indices = (labels == id_val).nonzero(as_tuple=True)[0]
                for i, j in combinations(id_indices.tolist(), 2):
                    if cam_ids_tensor[i] != cam_ids_tensor[j]:
                        cross_cam_pos += 1
            print(f"  Cross-camera positive pairs: {cross_cam_pos}")

        print("-" * 40)




