import torch
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(x_val, x_recon):
    x_val = torch.squeeze(x_val, dim=1)
    x_recon = torch.squeeze(x_recon, dim=1)
    x_val, x_recon = x_val.cpu().detach().numpy(), x_recon.cpu().detach().numpy()
    return ssim(x_val, x_recon, channel_axis=0, data_range=255)