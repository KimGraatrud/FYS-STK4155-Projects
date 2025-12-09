import os
from os import path
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image


class FacesDataset(Dataset):
    def __init__(self, images_dir, train=True):
        subdirs = sorted(os.listdir(images_dir))

        image_paths = []
        labels = []

        for i, label in enumerate(subdirs):
            filenames = sorted(os.listdir(path.join(images_dir, label)))
            for j, filename in enumerate(filenames):
                is_training_image = not (j % 10 == 9)
                if is_training_image == train:
                    image_paths.append(path.join(images_dir, label, filename))
                    labels.append(i)

        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        image = decode_image(img_path) / 255.0

        return image, label
