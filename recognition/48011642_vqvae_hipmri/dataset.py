"""
File: datset.py
Author: Jules de Beer
Last modified: 2025-10-26
Adapted from https://github.com/MishaLaskin/vqvae

Description: file for data loading, data preprocessing and creating data loaders for HipMRI dataset
"""

import cv2
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from os import listdir
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_data(batch_size, folder, transform, data_path):
    """Load HipMRI images and return data loader

    Args:
        batch_size (int): Batch size for data loader
        folder (str): Path of folder containing specific data split
        transform (transforms.Compose()): Transformations to apply to data
        data_path (str): Absolute path to folder containing folders of slices of dataset

    Returns:
        HipMRIDataset: Dataset for specified data split
        DataLoader: Dataloader for specified data split
        float: Variance of specified data split
    
    Example:
        > transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.28], std=[0.28])
            ])
        > data, loader, var = dataset.load_data(32, '/keras_slices_validate', transform,
        '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data')
    """

    data = HipMRIDataset(data_path + folder, transform=transform)

    loader = DataLoader(data,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True)
    
    var = np.var(data.data / 255.0)

    return data, loader, var

class HipMRIDataset(Dataset):
    """
    Custom Dataset class for HipMRI dataset
    """

    def __init__(self, path, transform=None):
        self.data = load_data_2D(path, listdir(path))
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)
        
        label = 0

        return img, label

    def __len__(self):
        return len(self.data)

def load_data_2D(path, image_names, dtype=np.float32):
    """Load 2D medical images from list of names and return a 3D array

    Args:
        path (str): Absolute path to list of images 
        image_names (str): List of image names
        dtype (np.type): Data type of returned list
    
    Returns:
        list: 3D array of 2D images

    Example:
        > path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'
        > image_names = list_dir(path)
        > images = load_data_2D(path, image_names)
    """

    num = len(image_names)

    # get fixed size for images based on first image
    first_case = nib.load(path + '/' + image_names[0]).get_fdata(caching='unchanged')
    # remove extra dimensions
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0]
    # make list for loaded images
    rows, cols = first_case.shape
    images = np.zeros((num, rows, cols), dtype=dtype)

    for i, name in enumerate(image_names):
        # load image
        niftiImage = nib.load(path + '/' + name)
        image = niftiImage.get_fdata(caching='unchanged') #read disk only
        # resize using interpolation
        if image.shape != (rows, cols):
            image = cv2.resize(image, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
        # remove extra dimensions
        if len(image.shape) == 3:
            image = image[:,:,0]
        # add to list
        images[i,:,:] = image.astype(dtype)

    return images
