import os
import math
import logging
import warnings

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize

from utils import initialize


# Load config files and env variables
initialize.load_logging()
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
        if size_limit:
            self.img_listing = img_listing[:size_limit]

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


def normalizing(torch_dataset):
    """
    TODO: standard normalize image dataset
    :param torch_dataset:
    :return:
    """
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

    return mu, std


def normalizing2(torch_dataset):
    """
    TODO: standard normalize image dataset
    :param torch_dataset:
    :return:
    """
    img_bulk = torch.stack([img_i for img_i, _ in torch_dataset], dim=3)
    logging.info("Finished building dataset image bulk")
    mu = img_bulk.type(torch.float32).view(3, -1).mean(dim=1)
    # returns: tensor([124.5226, 116.0461, 106.3240])
    std = img_bulk.type(torch.float32).view(3, -1).std(dim=1)
    # returns: tensor([66.6116, 64.9501, 65.5980])
    return mu, std


# cat_dogs_dataset = CatsDogsDataset(config.dataset.dataset_path, transform=transforms.v2.Resize((400, 400)))
# mu, std = normalizing2(cat_dogs_dataset)

img_transforms = transforms.Compose([
    transforms.v2.Resize((400, 400)),
    #transforms.v2.Resize((100, 100)),
    transforms.v2.ToDtype(torch.float32, scale=True),
    transforms.v2.ToTensor(),
    #transforms.v2.Normalize(mu, std)
])

cat_dogs_dataset = CatsDogsDataset(config.dataset.dataset_path, transform=img_transforms)
logging.info("cat_dogs ML dataset created")

# dataset_loader = torch.utils.data.DataLoader(cat_dogs_dataset, batch_size=4, shuffle=True, num_workers=0)
# logging.info("cat_dogs ML dataloader created")

