from torch.utils.data import Dataset
import h5py as h5
from src import utils
import numpy as np


class GalaxyDataset(Dataset):
    def __init__(self, mode="train"):
        """
        :param mode: 'train', 'test', or 'validate'
        """
        path = utils.DATA_PATHS[mode]

        self.file = h5.File(path, "r")

        self.images = self.file["image"]
        self.z = self.file["specz_redshift"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.z[index]

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

        # Convert to numpy and flatten
        data = np.array(data, dtype=np.float32)
        data = data.reshape(num_imgs, -1)

        return data
