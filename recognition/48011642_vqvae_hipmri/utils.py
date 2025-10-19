import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import time
import os
from dataset import HipMRIDataset
import numpy as np

def load_hipmri(test):
    data_folder_path = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'

    transform_train = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((128, 64)),
        transforms.Normalize(mean=[0.28], std=[0.28])
    ])

    transform_eval = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.28], std=[0.28])
    ])
    
    if test:
        val = HipMRIDataset(data_folder_path + '/keras_slices_test', train=False,
                        transform=transform_eval)
    else:
        val = HipMRIDataset(data_folder_path + '/keras_slices_validate', train=False,
                        transform=transform_eval)

    train = HipMRIDataset(data_folder_path + '/keras_slices_train', train=True,
                        transform=transform_train)
    
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(batch_size, test=False):    
    training_data, validation_data = load_hipmri(test)
    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size)
    x_train_var = np.var(training_data.data / 255.0)
    x_val_var = np.var(validation_data.data / 255.0)

    return training_data, validation_data, training_loader, validation_loader, x_train_var, x_val_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd()

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
