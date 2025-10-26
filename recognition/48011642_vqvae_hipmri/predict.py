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

parser = argparse.ArgumentParser()

parser.add_argument("--model_relative_path", type = str,
                    default = '/vqvae_data_sun_oct_19_22_46_34_2025.pth')
parser.add_argument("--image_relative_folder_path", type = int, default = '/keras_slices_test')
parser.add_argument("--n_predictions", type = int, default = 4)

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
data, data_loader, data_var = dataset.load_data(args.n_predictions, args.image_relative_folder_path, transform)

# Reconstruct data
(x, _) = next(iter(data_loader))
x = x.to(device)
model.eval()
with torch.no_grad():
    _, x_hat, _ = model(x)

# Display original images
utils.display_image_grid(x, 'original')

# Display reconstructed images with structural similarity index
diffs = []
labels = []
for i in range(args.n_predictions):
    x_t = torch.unsqueeze(x[i], dim = 0)
    x_r = torch.unsqueeze(x_hat[i], dim = 0)
    ssim, diff = utils.calculate_ssim(x_t, x_r)
    labels.append(round(ssim, 3))
    diffs.append(diff)
utils.display_image_grid(x_hat, 'prediction', labels)

# Display representation of differences between original and reconstructed images
utils.display_image_grid(diff, 'differences')