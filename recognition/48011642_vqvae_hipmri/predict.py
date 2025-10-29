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

parser.add_argument("--model_relative_path", type = str,
                    default = '/vqvae_data_sun_oct_19_22_46_34_2025.pth')
parser.add_argument("--data_path", type=str,
                    default='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data')
parser.add_argument("--image_folder", type = str, default = '/keras_slices_test')
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
    x_t = torch.unsqueeze(x[i], dim = 0)
    x_r = torch.unsqueeze(x_hat[i], dim = 0)
    ssim, diff = utils.calculate_ssim(x_t, x_r, full = True)
    labels.append(round(ssim, 3))
    diffs.append(diff)
utils.display_image_grid(args.n_predictions, x_hat, 'prediction', labels)

# Display representation of differences between original and reconstructed images
diffs = torch.from_numpy(np.array(diffs))
diffs = diffs.to(device)
utils.display_image_grid(args.n_predictions, diffs, 'differences')

params = vqvae_data['hyperparameters']

def generate_samples(e_indices):
    min_encodings = torch.zeros(e_indices.shape[0], params['n_embeddings']).to(device)
    min_encodings.scatter_(1, e_indices, 1)
    e_weights = model.vector_quantization.embedding.weight
    z_q = torch.matmul(min_encodings, e_weights).view((params["batch_size"],8,8,params["embedding_dim"])) 
    z_q = z_q.permute(0, 3, 1, 2).contiguous()

    x_recon = model.decoder(z_q)
    return x_recon, z_q,e_indices

import os
data_folder_path = os.getcwd() 
data_file_path = data_folder_path + '/latent_samples_100.npy'

samples = np.load(data_file_path,allow_pickle=True)

def reconstruct_from_pixelcnn(model,samples):
    

    min_encoding_indices = torch.tensor(samples).reshape(-1,1).long().to(device)
    x_recon, z_q,e_indices = generate_samples(min_encoding_indices)
    
    return x_recon, z_q,e_indices


x_val_recon,z_q,e_indices = reconstruct_from_pixelcnn(model,samples)

utils.display_image_grid(8, x_val_recon, 'generated.png')