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

    return cosine_distance


def customTripletMiner(features,labels):

    matrix = calcDistanceMatrix(features)

    positive_anchors = []
    positives = []
    negative_anchors = []
    negatives = []
    num_samples = labels.shape[0]

    # ....................................................... ŠEIT TURPINĀM

    # print("features:")
    # print(features)

    # print("Labels:")
    # print(labels)

    pass
