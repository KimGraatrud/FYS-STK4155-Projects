from torch.utils.data import Dataset
import h5py as h5
from src import utils
import numpy as np
import torch


class GalaxyDataset(Dataset):
    def __init__(self, mode="train", normalize=True):
        """
        :param mode: 'train', 'test', or 'validate'
        """
        path = utils.DATA_PATHS[mode]

        self.file = h5.File(path, "r")

        self.images = self.file["image"]
        self.z = self.file["specz_redshift"]

        self.normalize = normalize
        if normalize:
            norm = np.load(utils.NORM_URL)
            self.mean = np.expand_dims(norm["mean"], (1, 2))
            self.std = np.expand_dims(norm["std"], (1, 2))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        if self.normalize:
            img = (img - self.mean) / self.std

        img = torch.tensor(img, dtype=torch.float32)
        return img, self.z[index]

    def flat(self, num_imgs=None):
        """
        Return the first num_imgs images as a flat numpy array of shape (num_imgs, p),
        where p = 5*64*64.
        """
        # Limit to number of images loaded, if needed.
        num_imgs = (
            min(num_imgs, len(self.images))
            if num_imgs is not None
            else len(self.images)
        )

        # Load slice from the HDF5 dataset
        data = self.images[:num_imgs]

        # Convert to numpy
        data = np.array(data, dtype=np.float32)

        # Normalize
        if self.normalize:
            mu = np.expand_dims(self.mean, 0)
            sigma = np.expand_dims(self.std, 0)
            data = (data - mu) / sigma

        # Flatten
        data = data.reshape(num_imgs, -1)

        return data, np.array(self.z)

    def close(self):
        if self.file:
            self.file.close()
