import logging
import math
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize

from utils import initialize


logger = logging.getLogger(__name__)


# Load config files and env variables
config = initialize.load_config()
warnings.filterwarnings("ignore")


class CatsDogsDataset(Dataset):
    """
    Implements a custom PyTorch dataset for the cats and dogs images
    """
    def __init__(self, img_dir, transform, size_limit=None):
        self.img_dir = img_dir
        self.transform = transform

        dir_listing = os.listdir(self.img_dir)
        img_listing = [img_listing_i for img_listing_i in dir_listing if
                       (img_listing_i.endswith('.jpg') and
                       ((img_listing_i.rfind('cat') >= 0) or
                        (img_listing_i.rfind('dog') >= 0)))]

        self.img_listing = img_listing

    def __len__(self):
        return len(self.img_listing)

    def __getitem__(self, idx):
        path_idx = f'{self.img_dir}/{self.img_listing[idx]}'
        img_idx = read_image(path_idx)
        # Encode cat=0 and dog=1
        label_idx = torch.Tensor([0 if self.img_listing[idx].split('.')[0] == 'cat' else 1])

        if self.transform:
            img_idx = self.transform(img_idx)

        return img_idx, label_idx


def average_dimensions(torch_dataset):
    """
    Get the average W x H dimensions of the image dataset for uniform resizing
    :param torch_dataset:
    :return: average width, average height
    """
    avg_height, avg_width = 0.0, 0.0
    for images, _ in torch_dataset:
        num_channels, height, width = images.shape
        avg_height += height
        avg_width += width
    avg_height /= len(torch_dataset)
    avg_width /= len(torch_dataset)
    # round to the nearest 10th
    avg_height = math.ceil(avg_height / 10.0) * 10
    avg_width = math.ceil(avg_width / 10.0) * 10

    return avg_width, avg_height


def normalize_images(torch_dataset, method=1):
    """
    Get the mu and std for normalizing the dataset
    :param torch_dataset: pytorch dataset
    :param method: choose method 1 or 2 implementation
    :return: mean and standard deviation of full dataset
    """

    if method == 1:
        num_pixels, mu, std = 0.0, 0.0, 0.0
        for images, _ in torch_dataset:
            num_channels, height, width = images.shape
            num_pixels += height * width
            mu += images.numpy().mean(axis=(1, 2))
            std += images.numpy().std(axis=(1, 2))
            print(f"image i has RGB | : {mu} mean x {std} std")
        mu /= len(torch_dataset)
        # returns: array([124.52265245, 116.04606127, 106.32403549])
        std /= len(torch_dataset)
        # returns: array([58.01745754, 56.86948742, 56.91937481])

    elif method == 2:
        img_bulk = torch.stack([img_i for img_i, _ in torch_dataset], dim=3)
        logging.info("Finished building dataset image bulk")
        mu = img_bulk.type(torch.float32).view(3, -1).mean(dim=1)
        # returns: tensor([124.5226, 116.0461, 106.3240])
        std = img_bulk.type(torch.float32).view(3, -1).std(dim=1)
        # returns: tensor([66.6116, 64.9501, 65.5980])

    return mu, std


def tensor_transform(tensor_array, view_img=False):
    """
    Utility function to transform (and view) image sample into numpy format
    :param tensor_array:
    :return: new array in numpy format
    """
    minFrom, maxFrom = tensor_array.min(), tensor_array.max()
    minTo, maxTo = 0, 1
    tensor_array = minTo + (maxTo - minTo) * ((tensor_array - minFrom) / (maxFrom - minFrom))
    img_array = tensor_array.permute(1, 2, 0).numpy()
    if view_img:
        plt.imshow(img_array)
        plt.show()

    return img_array


img_transforms = transforms.Compose([
    transforms.v2.Resize((config.etl.image_resize, config.etl.image_resize)),
    transforms.v2.ToDtype(torch.float32, scale=True),
    transforms.v2.ToTensor(),
])

cat_dogs_dataset = CatsDogsDataset(config.dataset.dataset_path, transform=img_transforms)
logger.info("cat_dogs image dataset created")
