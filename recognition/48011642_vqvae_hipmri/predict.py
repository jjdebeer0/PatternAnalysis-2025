"""
File: predict.py
Author: Jules de Beer
Last modified: 2025-10-26
Adapted from https://github.com/MishaLaskin/vqvae

Description: file for predicting results and providing visualisations of trained model
"""

import torch
import utils
import dataset
import argparse
import torchvision.transforms.v2 as transforms
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--model_relative_path", type=str,
                    default='/vqvae_data_mon_oct_27_06_15_11_2025.pth')
parser.add_argument("--data_path", type=str,
                    default='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data')
parser.add_argument("--image_folder", type=str, default='/keras_slices_test')
parser.add_argument("--n_predictions", type=int, default=4)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved model
model, vqvae_data = utils.load_model(args.model_relative_path)

# Load data
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.28], std=[0.28])
    ])
data, data_loader, data_var = dataset.load_data(args.n_predictions, args.image_folder, transform,
                                                args.data_path)

# Reconstruct data
(x, _) = next(iter(data_loader))
x = x.to(device)
model.eval()
with torch.no_grad():
    _, x_hat, _ = model(x)

# Display original images
utils.display_image_grid(args.n_predictions, x, 'original')

# Display reconstructed images with structural similarity index
diffs = []
labels = []
for i in range(args.n_predictions):
    x_t = torch.unsqueeze(x[i], dim=0)
    x_r = torch.unsqueeze(x_hat[i], dim=0)
    ssim, diff = utils.calculate_ssim(x_t, x_r, full=True)
    labels.append(round(ssim, 3))
    diffs.append(diff)
utils.display_image_grid(args.n_predictions, x_hat, 'prediction', labels)

# Display representation of differences between original and reconstructed images
diffs = torch.from_numpy(np.array(diffs))
diffs = diffs.to(device)
utils.display_image_grid(args.n_predictions, diffs, 'differences')
