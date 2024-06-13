import torch

def euclidean_distance(x, y):
    """
    Compute Euclidean distance between two tensors.
    """
    return torch.pow(x - y, 2).sum(dim=1)
    #return torch.pow(x - y, 2).sum(dim=0)

def convert_to_triplets(a1, p, a2, n):

    # print("a1", a1)
    # print("p:", p)
    # print("Shapes")
    # print(a1.shape)
    # print(a2.shape)
    p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
    
    # print(p_idx, n_idx)
    # print(a1[p_idx], p[p_idx], n[n_idx])
    return a1[p_idx], p[p_idx], n[n_idx]


def CustomTripletLoss(ff, labels, hard_pairs, margin=0.3):

    anchor_indices, positive_indices, negative_anchor_indices, negative_indices = hard_pairs

    # ff - Feature vectors - [32,512]
    # labels - [32]
    # hard_pairs - list, anchor_indices, positive_indices ~ [6], negative_anchor_indices, negative_indices ~ [115]
    # print("FF")
    # print(ff)
    # print("LABELS")
    # print(labels)
    # print("HARD_PAIRS")
    # print(anchor_indices.shape, positive_indices, negative_anchor_indices.shape, negative_indices)

    a_idx, p_idx, n_idx = convert_to_triplets(anchor_indices, positive_indices, negative_anchor_indices, negative_indices)

    grad_fn_check = True if ff.grad_fn else False

    if len(a_idx) == 0:
        return torch.tensor(0.0, device=ff.device, requires_grad=grad_fn_check)
    

    # Gather the anchor, positive, and negative samples based on the hard pairs
    anchors = ff[a_idx]
    positives = ff[p_idx]
    negatives = ff[n_idx]

    # print(anchors.shape, positives.shape, negatives.shape)

    #Compute the distances
    pos_distances = euclidean_distance(anchors, positives)
    neg_distances = euclidean_distance(anchors, negatives)

    # Compute the triplet loss
    losses = torch.relu(pos_distances - neg_distances + margin)
    
    return torch.mean(losses)