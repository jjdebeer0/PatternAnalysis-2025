"""
File: modules.py
Author: Jules de Beer
Last modified: 2025-10-26
Adapted from https://github.com/MishaLaskin/vqvae

Description: file for components of vqvae including
    - ResidualLayer
    - ResidualStack
    - VectorQuantizer
    - Encoder
    - Decoder
    - VQVAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualLayer(nn.Module):
    """Custom Module class for ResidualLayer

    Args:
        in_dim (int): input dimensions
        h_dim (int): hidden layer dimensions
        res_h_dim (int): hidden dimension of residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()

        self.res_block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_dim, res_h_dim, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(res_h_dim),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
            nn.Conv2d(res_h_dim, h_dim, kernel_size = 1, stride = 1, bias = False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    """Custom module class of ResidualStack (stack of residual layers)

    Args:
        in_dim (int): input dimension
        h_dim (int): hidden layer dimension
        res_h_dim (int): hidden dimension of residual block
        n_res_layers (int): number of layers in stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

class VectorQuantizer(nn.Module):
    """Custom Module class for VectorQuantiser 

    Discretization bottleneck of the VQ-VAE.

    Args:
        n_e: number of embeddings
        e_dim: dimension of embedding
        beta: weight of commitment loss in loss term
            beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """Takes output of encoder, z, and maps it to discrete one-hot vector that is the index of
        the closest embedding vector e_j

        Args:
            z (tensor): continuous, 4D array of shape (B, C, H, W)
        
        Returns:
            float: embedding loss (k-clustering loss + commitment loss)
            tensor: z_q, discrete, 4D array of shape (B, C, H, W)
            float: perplexity, a measure of codebook usage (high perplexity, more indices used)
            tensor: closest encodings
            tensor: indices of closest encodings
        """

        # reshape and flattent z -> (B, C, H, W) -> (B*H*W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j, (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened**2, dim=1, keepdim = True) + \
            torch.sum(self.embedding.weight**2, dim = 1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim = 1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class Encoder(nn.Module):
    """Custom Module class for Encoder network.
    
    Args:
        in_dim (int): input dimension
        h_dim (int): hidden layer dimension
        res_h_dim (int): hidden dimension of residual block
        n_res_layers (int): number of layers in residual stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()

        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size = kernel, stride = stride, padding = 1),
            nn.BatchNorm2d(h_dim // 2),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size = kernel, stride = stride, padding=1),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(h_dim, h_dim, kernel_size = kernel-1, stride = stride-1, padding = 1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        """Given data sample x, maps it to the latent space, z, outputting parameters of a 
        categorical distribution"""

        return self.conv_stack(x)

class Decoder(nn.Module):
    """Custom Module class for Decoder network.

    Args:
        in_dim (int): input dimension
        h_dim (int): hidden layer dimension
        res_h_dim (int): hidden dimension of residual block
        n_res_layers (int): number of layers in residual stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()

        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size = kernel-1, stride = stride-1,
                               padding = 1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size = kernel, stride = stride,
                               padding = 1),
            nn.BatchNorm2d(h_dim // 2),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(h_dim//2, 1, kernel_size = kernel, stride = stride, padding = 1)
        )

    def forward(self, x):
        """Given a latent sample, z, maps it back to to the original space, x"""
        
        return self.inverse_conv_stack(x)

class VQVAE(nn.Module):
    """Custom Module class for VQVAE
    
    Args:
        h_dim (int): hidden layer dimension
        in_dim (int): input dimension
        res_h_dim (int): hidden dimension of residual block
        n_res_layers (int): number of layers in residual stack
        n_embeddings (int): number of embeddings
        embedding_dim (int): dimension of embedding
        beta (float): weight of commitment loss in loss term
        save_img_embedding_map (bool): If true, saves embedding map image
    """

    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size = 1, stride = 1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose = False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
