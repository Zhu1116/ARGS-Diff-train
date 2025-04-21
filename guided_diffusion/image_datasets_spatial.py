import random

import scipy.io
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_type,
    data_dir,
    batch_size,
    rank,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    image_path = data_dir + '/' + data_type + '.mat'
    dataset = ImageDataset(image_path, rank)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
        self,
        image_path,
        rank
    ):
        super().__init__()
        self.rank = rank
        self.hr_msi = scipy.io.loadmat(image_path)['HRMSI'] * 2 - 1  # load data and change the data range to -1~1
        self.resolution = self.hr_msi.shape[0]
        self.hr_msi = np.transpose(self.hr_msi, axes=(2, 0, 1)).astype(np.float32)

        self.channel_num = self.hr_msi.shape[0]
        self.imgs = []
        # crop img for training
        step = 32
        crop_size = 128
        self.crop_size = crop_size
        k = int((self.resolution - crop_size) / step) + 1
        self.k = k
        for i in range(k):
            for j in range(k):
                start_x = j * step
                start_y = i * step
                crop_img = self.hr_msi[:, start_y:start_y+crop_size, start_x:start_x+crop_size]
                self.imgs.append(crop_img)

    def __len__(self):
        return self.k * self.k

    def __getitem__(self, idx):
        img = self.imgs[idx]

        rank = self.rank
        output = np.zeros((rank, self.crop_size, self.crop_size), dtype=np.float32)
        for i in range(rank):
            output[i, :, :] = img[random.randint(0, self.channel_num-1), :, :]
        return output, {}