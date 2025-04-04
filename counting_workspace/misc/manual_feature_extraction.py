# Import the libraries
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from skimage.feature import local_binary_pattern

from skimage.transform import resize
## Preparing Image Data for Assignment 1
import numpy as np
from skimage.color import rgb2hsv
#import matplotlib.pyplot as plt
#import pandas as pd
#from skimage.color import rgb2gray
#from skimage.transform import resize
#from skimage.feature import hog
#from skimage.feature import local_binary_pattern
#import os

#----------------------------------------------------------------
def fox_get_colour_features(img, fstr = 'RGB', blocks_r = 1,\
     blocks_c = 1, bins = 20):
    # Returns a set of features extracted from an image.   
    # Five different feature types are encoded.
    # Input −−−−−      
    # 'img': RGB input image      
    # 'fstr': feature string − one of these − RGB, 
    #                   HSV, CHR, OHT, H (default = 'RGB')
    # 'blocks_r': number of rows of blocks to split 
    #               the image into (default = 1)
    # 'blocks_c': number of columns of blocks to split 
    #               the image into (default = 1)
    # 'bins': number of bins for histogram methods
    # 
    # Calculates rgb averages and stds of the blocks in the 
    # image and concatenates them. For example, for RGB, the
    # row of features is arranged as:
    # [ra1, ga1, ba1, rs1, gs1, bs1, ra2, ga2, ba2, rs2, ...]
    # where "a" stands for average, "s" is for std, and the
    # number is the block number counted row by row.
    #
    # For example, for blocks_r = 2, blocks_c = 3, the blocks are
    # labelled as:
    # [1 2 3]
    # [4 5 6]
    #
    # For the RGB example, the total number of features is
    # 2(rows)x3(cols)x6(features) = 36 features
    
    # Split the image
    imsz = img.shape
    
    indrows = np.linspace(1,imsz[0],blocks_r+1).astype(int)
    indcols = np.linspace(1,imsz[1],blocks_c+1).astype(int)

    feature_values = None

    if len(imsz) == 2:
        # grey image
        print('RGB image is expected')
        return []

    for i in range(blocks_r):
        for j in range(blocks_c):
            if fstr == 'H':
                h,be = colour_features_histogram(img,bins)
                if feature_values is None:
                    feature_values = h
                else:
                    feature_values = np.append(feature_values,h)
        
            else:
                imblock = img[indrows[i]:indrows[i+1],\
                              indcols[j]:indcols[j+1],:]
                x = colour_features_means(imblock,fstr)
                if feature_values is None:
                    feature_values = x
                else:
                    feature_values = np.append(feature_values,x)

    return feature_values
#-------------------------------------------------------------------

def colour_features_means(im,colour_space):
    
    if colour_space == 'RGB':
        r = im[:,:,0]; g = im[:,:,1]; b = im[:,:,2] # split into 3 panels
        cols = np.hstack((np.reshape(r,(-1,1)), np.reshape(g,(-1,1)), \
            np.reshape(b,(-1,1))))
    elif colour_space == 'HSV':
        im = rgb2hsv(im)
        r = im[:,:,0]; g = im[:,:,1]; b = im[:,:,2] # split into 3 panels
        cols = np.hstack((np.reshape(r,(-1,1)), np.reshape(g,(-1,1)), \
            np.reshape(b,(-1,1))))
    elif colour_space == 'CHR': #chrominance
        im = np.array(im).astype(float)
        d = np.sqrt(np.sum(im**2,axis=2)) + 0.001
        # guard against division by zero
        c1 = im[:,:,0]/d; c2 = im[:,:,1]/d
        cols = np.hstack((np.reshape(c1,(-1,1)), np.reshape(c2,(-1,1))))
    elif colour_space == 'OHT': # Ohta 1980 space
        im = np.array(im).astype(float)
        r = im[:,:,0]; g = im[:,:,1]; b = im[:,:,2] # split into 3 panels
        i1 = (r + g + b) / 3
        i2 = r - b
        i3 = (2*g - r - b) / 2
        cols = np.hstack((np.reshape(i1,(-1,1)), np.reshape(i2,(-1,1)), \
            np.reshape(i3,(-1,1))))
    else:
        print('Unknown feature space')
        return []

    x = np.hstack((np.mean(cols,axis = 0), np.std(cols,axis = 0)))
    return x
#-------------------------------------------------------------------

def colour_features_histogram(im,bins):
    im = rgb2hsv(im)
    h = np.reshape(im[:,:,0],(-1,1))
    hist, bin_edges = np.histogram(h,bins = bins, range = [0,1])
    hist = hist/np.sum(hist)
    return hist, bin_edges

def extract_manual_features(img, r=128, feature_type=None):
    """
    Extracts manual features from an image.
    
    Parameters:
    - img: Input image
    - r: Resize dimension (default: 224)
    - feature_type: Specifies which feature to extract ("HOG", "LBP", "RGB", "H10").
                    If None, all features are extracted and concatenated.

    Returns:
    - Extracted feature array (numpy array)
    """
    if feature_type is 'Combined':
        feature_type = None
    # Resize the image
    resized_img = resize(img, (r, r))
    
    extracted_features = []

    # HOG Feature Extraction
    if feature_type is None or feature_type == "HOG":
        fd = hog(resized_img, orientations=9, pixels_per_cell=(6, 6),
                 cells_per_block=(3, 3), channel_axis=2)
        extracted_features.append(fd)

    # LBP Feature Extraction
    if feature_type is None or feature_type == "LBP" or feature_type == "LBP+H10":
        radius = 3
        n_points = 8 * radius
        method = 'uniform'
        lbp_raw = local_binary_pattern(rgb2gray(img), n_points, radius, method)
        lbp, _ = np.histogram(lbp_raw, bins=64, density=True)
        extracted_features.append(lbp)

    # RGB Block Features
    if feature_type is None or feature_type == "RGB":
        rgb = fox_get_colour_features(img, fstr='RGB', blocks_r=8, blocks_c=8, bins=16)
        extracted_features.append(rgb)

    # H10 Features
    if feature_type is None or feature_type == "H10" or feature_type == "LBP+H10":
        h10 = fox_get_colour_features(img, fstr='H', blocks_r=2, blocks_c=2, bins=32)
        extracted_features.append(h10)

    # Return selected feature or concatenated features
    return np.concatenate(extracted_features, axis=0) if extracted_features else None