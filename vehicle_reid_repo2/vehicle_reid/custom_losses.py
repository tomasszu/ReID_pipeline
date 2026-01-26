import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenWorldBatchLoss(nn.Module):
    """
    Open-World Batch Separation Loss
    Optimizes separation between hard positives and hard negatives
    using similarity distribution tails (quantiles).
    """

    def __init__(self, margin=0.1, pos_q=0.1, neg_q=0.9, eps=1e-8):
        super().__init__()
        self.margin = margin
        self.pos_q = pos_q
        self.neg_q = neg_q
        self.eps = eps

    def forward(self, embeddings, labels):
        """
        embeddings: Tensor [B, D] (assumed L2-normalized)
        labels:     Tensor [B]
        """

        B = embeddings.size(0)
        device = embeddings.device

        # cosine similarity matrix
        sim = embeddings @ embeddings.t()

        # mask self-similarity
        eye = torch.eye(B, device=device, dtype=torch.bool)

        labels_eq = labels[:, None] == labels[None, :]

        pos_mask = labels_eq & ~eye
        neg_mask = ~labels_eq

        # handle degenerate batches safely
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)

        pos_sim = sim[pos_mask]
        neg_sim = sim[neg_mask]

        # hard statistics (distribution tails)
        pos_hard = torch.quantile(pos_sim, self.pos_q)
        neg_hard = torch.quantile(neg_sim, self.neg_q)

        # margin-based separation loss
        loss = F.relu(self.margin + neg_hard - pos_hard)

        return loss
    
class CenterBasedEmbeddingLoss(nn.Module):
    """
    Center-based Embedding Loss
    Optimizes separation between class centers using cosine similarity.
    """

    def __init__(self, scale=32):
        super().__init__()
        self.scale = scale

    def forward(self, embeddings, labels, class_centers):
        """
        embeddings:     Tensor [B, D] (assumed L2-normalized)
        labels:         Tensor [B]
        class_centers: Tensor [C, D] (assumed L2-normalized)
        """

        logits_centers = self.scale * (embeddings @ class_centers.t())

        loss = F.cross_entropy(logits_centers, labels)

        return loss
