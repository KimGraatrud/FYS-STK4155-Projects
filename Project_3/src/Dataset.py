from torch.utils.data import Dataset
import h5py as h5
from src import utils


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
