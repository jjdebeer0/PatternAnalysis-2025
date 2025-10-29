"""
File: train.py
Author: Jules de Beer
Last modified: 2025-10-26
Adapted from https://github.com/MishaLaskin/vqvae

Description: file for training, validating, testing and saving VQVAE
"""

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import argparse
import utils
import dataset
from modules import VQVAE

parser = argparse.ArgumentParser()

# Hyper parameters
timestamp = utils.readable_timestamp()

parser.add_argument("--data_path", type=str,
                    default='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=80000)
parser.add_argument("--n_hiddens", type=int, default=256)
parser.add_argument("--n_residual_hiddens", type=int, default=64)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=32)
parser.add_argument("--n_embeddings", type=int, default=256)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-1)
parser.add_argument("--log_interval", type=int, default=50)

parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in /vqvae_' + args.filename + '.pth')

# Define data transforms and load data splits with data loaders from dataset.py
train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((128, 64)),
    transforms.Normalize(mean=[0.28], std=[0.28])
    ])

eval_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.28], std=[0.28])
    ])

train_data, train_loader, train_var = dataset.load_data(args.batch_size, '/keras_slices_train',
                                                        train_transform, args.data_path)
val_data, val_loader, val_var = dataset.load_data(args.batch_size, '/keras_slices_validate',
                                                  eval_transform, args.data_path)
test_data, test_loader, test_var = dataset.load_data(540, '/keras_slices_test', eval_transform,
                                                     args.data_path)

# Set up VQVAE model with components from modules.py
model = VQVAE(args.n_hiddens, args.n_residual_hiddens, args.n_residual_layers, args.n_embeddings,
              args.embedding_dim, args.beta).to(device)

def train():
    """Train model, including a validation step"""

    print("> Training")

    # Set up optimiser and leanring rate scheduler
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, amsgrad = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0 = 5000, eta_min = 3e-5
    )

    # Set up dictionary for tracking metrics during training
    metrics = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
        'val_loss_vals': [],
        'val_recon_errors': [],
        'ssim': []
    }

    # Training loop
    for i in range(args.n_updates):
        model.train()
        (x, _) = next(iter(train_loader))
        x = x.to(device)
        optimizer.zero_grad()

        # Calculate training loss
        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Save training metrics
        metrics["recon_errors"].append(recon_loss.cpu().detach().numpy())
        metrics["perplexities"].append(perplexity.cpu().detach().numpy())
        metrics["loss_vals"].append(loss.cpu().detach().numpy())
        metrics["n_updates"] = i

        # Validation
        model.eval()
        with torch.no_grad():
            (v, _) = next(iter(val_loader))
            v = v.to(device)

            # Calculate validation loss
            val_embedding_loss, v_hat, _ = model(v)
            val_recon_loss = torch.mean((v_hat - v)**2) / val_var
            val_loss = val_recon_loss + val_embedding_loss

            # Calculate structural similarity between original and reconstructed images
            ssim = utils.calculate_ssim(v, v_hat)

            # Save validation metrics
            metrics["val_recon_errors"].append(val_recon_loss.cpu().detach().numpy())
            metrics["val_loss_vals"].append(val_loss.cpu().detach().numpy())
            metrics["ssim"].append(ssim)

        # Save model and print metrics at regular intervals during training
        if i % args.log_interval == 0:

            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(model, metrics, hyperparameters, args.filename)

            print('Update #', i, 
                  'Recon Loss:', np.mean(metrics["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(metrics["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(metrics["perplexities"][-args.log_interval:]),
                  'SSIM:', np.mean(metrics["ssim"][-args.log_interval:]))
    
    utils.plot_metrics(metrics)

def test():
    """Test model and report structural similarity index accuracy score"""

    print("> Testing")
    
    # Set up dictionary for tracking metrics during testing
    metrics = {
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
        'ssim': []
    }

    # Test
    model.eval()
    with torch.no_grad():
        (x, _) = next(iter(test_loader))

        x = x.to(device)
        embedding_loss, x_hat, perplexity = model(x)

        # Calculate test loss
        recon_loss = torch.mean((x_hat - x)**2) / test_var
        loss = recon_loss + embedding_loss
        # Calculate structural similarity between original and reconstructed images
        ssim = utils.calculate_ssim(x, x_hat)

        # Save testing metrics
        metrics["ssim"].append(ssim)            
        metrics["recon_errors"].append(recon_loss.cpu().detach().numpy())
        metrics["perplexities"].append(perplexity.cpu().detach().numpy())
        metrics["loss_vals"].append(loss.cpu().detach().numpy())   
    
    # Print overall metrics
    print('Recon Loss:', np.mean(metrics["recon_errors"]),
          'Loss', np.mean(metrics["loss_vals"]),
          'Perplexity:', np.mean(metrics["perplexities"]),
          'SSIM:', np.mean(metrics["ssim"]))

if __name__ == "__main__":
    train()
    test()