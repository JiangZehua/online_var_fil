import glob
from ntpath import join
import os
from matplotlib import pyplot as plt
import numpy as np
from pdb import set_trace as TT
from PIL import Image

import torch

from vae import Model, MyData\
    , unnormalize

# def unnormalize(x, mean_path, std_path):
#         mean = np.load(mean_path)
#         std = np.load(std_path)
#         TT()
#         return mean + x * std


if __name__ == '__main__':
    sample_path = "datasets/test1"
    path_list = [] 
    # select 5 pictures in sample_train_data_path
    for i in range(100, 106):
        s_path = os.path.join(sample_path, f"image_{i}.png")
        path_list.append(s_path)
    
    # plot the 5 pictures in path_list
    for i in range(5):
        img = Image.open(path_list[i])
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
    # plt.show()

    data_path = "outputs/2022-04-28_11-55-39_dmlab_run/"
    # mean_path = join(data_path, "q_means.npy")
    # std_path = join(data_path, "q_stds.npy")

    q_means = np.load(join(data_path, "q_means.npy"))
    q_stds = np.load(join(data_path, "q_stds.npy"))
    dataset = MyData(r'datasets/test1')

    model = Model().cuda()

    z = q_means[100:106, :]
    y = q_stds[100:106, :]
    z = torch.from_numpy(z).cuda()
    y = torch.from_numpy(y).cuda()

    # image_batch = model.decoder(z)
    image_batch = model.decoder(y)

    fig, ax = plt.subplots(1, 5)
    for j in range(5):
        ax[j].imshow(
            unnormalize(image_batch[j, :, :, :], dataset)\
                .transpose(0,1).transpose(1,2).cpu().detach().numpy()
            # unnormalize(image_batch[j, :, :, :], mean_path, std_path)\
            #     .transpose(0,1).transpose(1,2).cpu().detach().numpy()
            
            # image_batch.permute(0, 2, 3, 1)[j, :, :, :].cpu().detach().numpy()
    )
    plt.show()



    