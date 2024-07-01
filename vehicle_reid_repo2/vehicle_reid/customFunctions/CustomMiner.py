import torch
import numpy as np

def calcDistanceMatrix(ff):

    if ff.is_cuda:
        features = ff.cpu()

    features = features.detach().numpy()
    
    # Normalize the features
    norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Compute the cosine similarity matrix
    cosine_similarity = np.dot(norm_features, norm_features.T)

    # Convert cosine similarity to cosine distance
    cosine_distance = 1 - cosine_similarity

    # Convert the numpy array back to a PyTorch tensor
    cosine_distance_tensor = torch.tensor(cosine_distance, device=ff.device, dtype=ff.dtype)

    return cosine_distance_tensor

def get_all_pairs(matrix, labels):

    labels1 = labels.unsqueeze(1)
    labels2 = labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    matches.fill_diagonal_(0)

    return matches, diffs

def customTripletMiner(features,labels, epsilon=0.1):

    matrix = calcDistanceMatrix(features)

    matches, diffs = get_all_pairs(matrix, labels)

    pa_idx, p_idx = torch.where(matches)
    na_idx, n_idx = torch.where(diffs)

    if len(pa_idx) == 0 or len(na_idx) == 0:
        empty = torch.tensor([], device=labels.device, dtype=torch.long)
        return empty.clone(), empty.clone(), empty.clone(), empty.clone()
    
    mat_neg_sorting = matrix
    mat_pos_sorting = matrix.clone()

    dtype = matrix.dtype

    pos_ignore = torch.finfo(dtype).min  # Negative infinity
    neg_ignore = torch.finfo(dtype).max  # Positive infinity

    mat_pos_sorting[na_idx, n_idx] = pos_ignore
    mat_neg_sorting[pa_idx, p_idx] = neg_ignore

    mat_pos_sorting.fill_diagonal_(pos_ignore)
    mat_neg_sorting.fill_diagonal_(neg_ignore)

    pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
    neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

    hard_pos_idx = torch.where(
        pos_sorted + epsilon > neg_sorted[:, 0].unsqueeze(1)
    )
    hard_neg_idx = torch.where(
        neg_sorted - epsilon < pos_sorted[:, -1].unsqueeze(1)
    )


    positive_anchors = hard_pos_idx[0]
    positives = pos_sorted_idx[positive_anchors, hard_pos_idx[1]]
    negative_anchors = hard_neg_idx[0]
    negatives = neg_sorted_idx[negative_anchors, hard_neg_idx[1]]

    # # print("features:")
    # # print(features)

    # print("Labels:")
    # print(labels)

    # # print("positive sorting:")
    # # print(pos_sorted)

    # print("hard_pos_idx:")
    # print(hard_pos_idx)

    # print("hard_neg_idx:")
    # print(hard_neg_idx)

    # pass

    return positive_anchors, positives, negative_anchors, negatives
