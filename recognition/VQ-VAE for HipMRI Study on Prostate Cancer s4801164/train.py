"""
File: train.py
Author: Jules de Beer
Last modified: 2025-10-30
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
parser.add_argument("--n_epochs", type=int, default=700)
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
    transforms.Normalize(mean=[0.28], std=[0.28]),
    transforms.RandomResizedCrop(size=(128, 64), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((128, 64))
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0 = 5000, eta_min = 3e-5
    )

    # Set up dictionary for tracking metrics during training
    metrics = {
        'n_epochs': 0,
        'train_recon_errors': [0],
        'train_loss': [0],
        'train_ssim' : [0],
        'train_perplexities': [0],
        'val_loss': [0],
        'val_recon_errors': [0],
        'val_ssim': [0],
        'val_perplexities': [0]
    }

    # Training loop
    for epoch in range(args.n_epochs):
        if metrics["val_ssim"][epoch] > 0.65:
            break

        model.train()

        train_running_recon_error = 0
        train_running_perplexity = 0
        train_running_loss = 0
        train_running_ssim = 0
        for idx, batch in enumerate(train_loader):
            x = batch[0].to(device)
            optimizer.zero_grad()

            # Calculate training loss
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / train_var
            loss = recon_loss + embedding_loss
            ssim = utils.calculate_ssim(x, x_hat)

            train_running_recon_error += recon_loss
            train_running_perplexity += perplexity
            train_running_loss += loss
            train_running_ssim += ssim

            loss.backward()
            optimizer.step()
            scheduler.step()
        
        train_running_recon_error = train_running_recon_error / (idx + 1)
        train_running_perplexity = train_running_perplexity / (idx + 1)
        train_running_loss = train_running_loss / (idx + 1)
        train_running_ssim = train_running_ssim / (idx + 1)

        # Save training metrics
        metrics["train_recon_errors"].append(train_running_recon_error.cpu().detach().numpy())
        metrics["train_perplexities"].append(train_running_perplexity.cpu().detach().numpy())
        metrics["train_loss"].append(train_running_loss.cpu().detach().numpy())
        metrics["train_ssim"].append(ssim)

        metrics["n_epoch"] = epoch

        # Validation
        val_running_recon_error = 0
        val_running_perplexity = 0
        val_running_loss = 0
        val_running_ssim = 0
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                v = batch[0].to(device)

                # Calculate validation loss
                val_embedding_loss, v_hat, val_perplexity = model(v)
                val_recon_loss = torch.mean((v_hat - v)**2) / val_var
                val_loss = val_recon_loss + val_embedding_loss
                # Calculate structural similarity between original and reconstructed images
                val_ssim = utils.calculate_ssim(v, v_hat)

                val_running_recon_error += val_recon_loss
                val_running_perplexity += val_perplexity
                val_running_loss += val_loss
                val_running_ssim += val_ssim

            val_running_recon_error = val_running_recon_error / (idx + 1)
            val_running_perplexity = val_running_perplexity / (idx + 1)
            val_running_loss = val_running_loss / (idx + 1)
            val_running_ssim = val_running_ssim / (idx + 1)

            # Save validation metrics
            metrics["val_recon_errors"].append(val_running_recon_error.cpu().detach().numpy())
            metrics["val_loss"].append(val_running_loss.cpu().detach().numpy())
            metrics["val_perplexities"].append(val_running_perplexity.cpu().detach().numpy())
            metrics["val_ssim"].append(val_running_ssim)

        # Save model and print metrics at regular intervals during training
        if epoch % args.log_interval == 0:

            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(model, metrics, hyperparameters, args.filename)

            print('Epoch: ', epoch, 
                'Train recon Loss:', np.mean(metrics["train_recon_errors"][-args.log_interval:]),
                'Train loss:', np.mean(metrics["train_loss"][-args.log_interval:]),
                'Train perplexity:', np.mean(metrics["train_perplexities"][-args.log_interval:]),
                'Train SSIM: ', np.mean(metrics["train_ssim"][-args.log_interval:]),
                'Val recon Loss:', np.mean(metrics["val_recon_errors"][-args.log_interval:]),
                'Val loss:', np.mean(metrics["val_loss"][-args.log_interval:]),
                'Val perplexity:', np.mean(metrics["val_perplexities"][-args.log_interval:]),
                'Val SSIM: ', np.mean(metrics["val_ssim"][-args.log_interval:])
                )
    
    utils.plot_metrics(metrics)

def test():
    """Test model and report structural similarity index accuracy score"""

    print("> Testing")
    
    # Set up dictionary for tracking metrics during testing
    metrics = {
        'recon_errors': [],
        'loss': [],
        'perplexities': [],
        'ssim': []
    }

    # Test
    model.eval()
    test_running_recon_error = 0
    test_running_perplexity = 0
    test_running_loss = 0
    test_running_ssim = 0
    with torch.no_grad():

        for idx, batch in enumerate(test_loader):
            x = batch[0].to(device)

            embedding_loss, x_hat, perplexity = model(x)
            # Calculate test loss
            recon_loss = torch.mean((x_hat - x)**2) / test_var
            loss = recon_loss + embedding_loss
            # Calculate structural similarity between original and reconstructed images
            ssim = utils.calculate_ssim(x, x_hat)

            test_running_recon_error += recon_loss
            test_running_loss += loss
            test_running_perplexity += perplexity
            test_running_ssim += ssim
        
    test_running_recon_error = test_running_recon_error / (idx + 1)
    test_running_perplexity = test_running_perplexity / (idx + 1)
    test_running_loss = test_running_loss / (idx + 1)
    test_running_ssim = test_running_ssim / (idx + 1)

    # Save testing metrics
    metrics["ssim"].append(test_running_ssim)            
    metrics["recon_errors"].append(test_running_recon_error.cpu().detach().numpy())
    metrics["perplexities"].append(test_running_perplexity.cpu().detach().numpy())
    metrics["loss"].append(test_running_loss.cpu().detach().numpy())
    
    # Print overall metrics
    print('Recon Loss:', np.mean(metrics["recon_errors"]),
          'Loss', np.mean(metrics["loss"]),
          'Perplexity:', np.mean(metrics["perplexities"]),
          'SSIM:', np.mean(metrics["ssim"]))

if __name__ == "__main__":
    train()
    test()