"""
File: utils.py
Author: Jules de Beer
Last modified: 2025-10-26
Adapted from https://github.com/MishaLaskin/vqvae

Description: file for utilities
"""

import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from modules import VQVAE
from skimage.metrics import structural_similarity as ssim
from scipy.signal import savgol_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_ssim(x, x_hat, full = False):
    """Calculate structural similarity index for paired image and reconstruction

    Args:
        x (Tensor): original images
        x_hat (Tensor): reconstructed images
    
    Returns:
        tuple: the SSIM value ranging from -1 (inverse correlation) to 1 (perfect correlation)
        and an image (array) highlighting differences between the two input images
    
    Examples:
        > x = x.to(device)
        > _, x_hat, _ = model(x)
        > ssim, diff = calculate_ssim(x, x_hat)
    """

    # reshape to (B, H, W) (grayscale image only has one channel anyway)
    x = torch.squeeze(x, dim = 1)
    x_hat = torch.squeeze(x_hat, dim = 1)

    x, x_hat = x.cpu().detach().numpy(), x_hat.cpu().detach().numpy()
    return ssim(x, x_hat, channel_axis = 0, data_range = 255, full = full)

def readable_timestamp():
    """Format timestamp
    
    Returns:
        str: formatted timestamp
    
    Examples:
        > timestamp = readable_timestamp(timestamp)
    """
    return time.ctime().replace('  ', ' ').replace(' ', '_').replace(':', '_').lower()

def save_model_and_results(model, metrics, hyperparameters, timestamp):
    """Save model, metrics and hyperparameters to current directory

    Args:
        model (VQVAE): VQVAE model
        metrics (dict): dictionary of metrics over the course of training
        hyperparameters (dict): dictionary of model hyperparameters
        timestamp (str): current time
    
    Examples:
        > save_model_and_results(model, metrics, hyperparameters, timestamp)
    """

    path = os.getcwd()

    # results saved as a dictionary
    results_to_save = {
        'model': model.state_dict(),
        'metrics': metrics,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, path + '/vqvae_data_' + timestamp + '.pth')

def load_model(model_filename):
    """Load model, metrics and hyperparameters from current directory

    Args:
        model_filename (str): model file name

    Returns:
        VQVAE: VQVAE model
        dict: dictionary of model data
    
    Examples:
        > model, data = load_model(relative_path)
    """

    path = os.getcwd()

    # Load dictionary
    if torch.cuda.is_available():
        data = torch.load(path + model_filename, weights_only = False)
    else:
        data = torch.load(path+model_filename, map_location = lambda storage, loc: storage,
                          weights_only = False)

    # Unpack hyperparameters from dictionary
    params = data["hyperparameters"]

    # Instantiate model with appropriate hyperparameters
    model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                  params['n_residual_layers'], params['n_embeddings'],
                  params['embedding_dim'], params['beta']).to(device)

    # Load into model the state dictionary
    model.load_state_dict(data['model'])

    return model, data

def plot_metrics(metrics):
    """Plot training and validation metrics of a model and saves to current directory

    Args:
        metrics (dict): dictionary of metrics over the course of training

    Examples:
        > plot_metrics(metrics)
    """

    # Retrieve time course lists for each metric
    recon_errors = savgol_filter(metrics["recon_errors"], 19, 5)
    perplexities = savgol_filter(metrics["perplexities"], 19, 5)
    loss_vals = savgol_filter(metrics["loss_vals"], 19, 5)
    val_recon_errors = savgol_filter(metrics["val_recon_errors"], 19, 5)
    val_loss = savgol_filter(metrics["val_loss_vals"], 19, 5)
    ssim = savgol_filter(metrics["ssim"], 19, 5)

    # Reconstruction loss (training and validation) over time
    f = plt.figure(figsize = (16,4))
    ax = f.add_subplot(1,4,2)
    ax.plot(recon_errors, label = 'training')
    ax.plot(val_recon_errors, label = 'validation')
    ax.set_yscale('log')
    ax.set_title('Reconstruction Loss')
    ax.set_xlabel('Iteration')

    # Perplexity (training) over time
    ax = f.add_subplot(1,4,4)
    ax.plot(perplexities)
    ax.set_title('Average codebook usage (perplexity).')
    ax.set_xlabel('Iteration')

    # Loss (training and validation) over time
    ax = f.add_subplot(1,4,1)
    ax.plot(loss_vals, label = 'training')
    ax.plot(val_loss, label = 'validation')
    ax.set_yscale('log')
    ax.set_title('Overall Loss')
    ax.set_xlabel('Iteration')

    # Structural similarity index (validation) over time
    ax = f.add_subplot(1,4,3)
    ax.plot(ssim)
    ax.set_title('SSIM')
    ax.set_xlabel('Iteration')

    ax.legend()

    plt.savefig("metrics.png")
    plt.close()

def display_image_grid(num_images, images, name, labels = None):
    """Generates grid of images for given images and saves to current directory

    Args:
        num_images (int): number of images to display
        images (Tensor): 2D grayscale images
        name (str): name of the plot
        labels (list): list of labels for each image

    Examples:
        > x = x.to(device)
        > display_image_grid(x, 'original')
    """

    images = images.cpu().detach() + 0.5
    images = images.numpy()

    fig = plt.figure(figsize = (8, 8))
    cols = 4
    rows = num_images // cols if num_images % cols == 0 else num_images // cols + 1

    for i in range(1, num_images + 1):

        label = ""
        if labels is not None:
            label = labels[i - 1]
        
        ax = fig.add_subplot(rows, cols, i)
        ax.set_title(label)
        plt.imshow(np.transpose(images[i - 1], (1, 2, 0)), interpolation='nearest', cmap='gray')
        plt.axis('off')
    
    plt.savefig(f'{name}.png')
