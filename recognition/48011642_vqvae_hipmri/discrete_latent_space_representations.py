import torch
import utils
import dataset
import argparse
import torchvision.transforms.v2 as transforms
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--model_relative_path", type = str,
                    default = '/vqvae_data_mon_oct_27_06_15_11_2025.pth')
parser.add_argument("--data_path", type=str,
                    default='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data')
parser.add_argument("--train_folder", type = str, default = '/keras_slices_train')
parser.add_argument("--test_folder", type = str, default = '/keras_slices_test')

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
train_data, train_loader, train_var = dataset.load_data(1, args.train_folder, transform,
                                                args.data_path)
test_data, test_loader, test_var = dataset.load_data(1, args.test_folder, transform,
                                                args.data_path)
def encode(data_loader):
    tot_e_indices = torch.empty([0, 32, 64])
    tot_e_indices = tot_e_indices.to(device)
    for _, data in enumerate(data_loader):
        (x, _) = data
        x = x.to(device)
        vq_encoder_output = model.pre_quantization_conv(model.encoder(x))
        _, z_q, _, _,e_indices = model.vector_quantization(vq_encoder_output)
        e_indices = e_indices.to(device)
        e_indices = torch.reshape(e_indices, (32, 64))
        e_indices = torch.unsqueeze(e_indices, 0)
        tot_e_indices = torch.cat((tot_e_indices, e_indices), dim=0)
    print(tot_e_indices.shape)
    return e_indices

np.save('latent_e_indices_train.npy', encode(train_loader).cpu().detach().numpy())
np.save('latent_e_indices_test.npy', encode(test_loader).cpu().detach().numpy())