import cv2
import numpy as np
from torch.utils.data import Dataset
from os import listdir
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt


class HipMRIDataset(Dataset):
    """
    Creates block dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None):
        image_names = listdir(file_path)
        data = load_data_2D(file_path, image_names)
        self.data = np.expand_dims(data, axis=1)

        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)

import nibabel as nib

def to_channels(arr: np.ndarray, dtype=np.uint8)-> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + ( len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

#load medical image functions
def load_data_2D(folder_path, imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory
    usage.
    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick
    loading and testing scripts.
    '''
    affines = []

    #get fixed size
    num = len(imageNames)
    first_case = nib.load(folder_path + '/' + imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] #sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(imageNames):
        niftiImage = nib.load(folder_path + '/' + inName)
        inImage = niftiImage.get_fdata(caching='unchanged') #read disk only
        affine = niftiImage.affine
        if inImage.shape != (rows, cols):
            inImage = cv2.resize(inImage, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] #sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage- inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images