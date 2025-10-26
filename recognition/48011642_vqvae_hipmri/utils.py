import torch
import time
import os
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(x_val, x_recon):
    x_val = torch.squeeze(x_val, dim=1)
    x_recon = torch.squeeze(x_recon, dim=1)
    x_val, x_recon = x_val.cpu().detach().numpy(), x_recon.cpu().detach().numpy()
    return ssim(x_val, x_recon, channel_axis=0, data_range=255)

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
