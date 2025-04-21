import random
import scipy.io
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


def load_data(
    *,
    data_type,
    data_dir,
    batch_size,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    image_path = data_dir + '/' + data_type + '_gaussian.mat'
    dataset = ImageDataset(image_path)
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
    ):
        super().__init__()
        self.img = scipy.io.loadmat(image_path)['LRHSI'].astype(np.float32) * 2 - 1  # load data and change the data range to -1~1
        self.resolution = self.img.shape[0]

    def __len__(self):
        return self.resolution * self.resolution

    def __getitem__(self, idx):
        row = idx // self.resolution
        col = idx % self.resolution

        spectral_values = self.img[row, col, :]

        return spectral_values, {}

    # def __getitem__(self, idx):
    #     rank = 5
    #     band = self.img.shape[2]
    #     output = np.zeros((rank, band), dtype=np.float32)
    #     for i in range(rank):
    #         idx = random.randint(0, self.resolution * self.resolution - 1)
    #         row = idx // self.resolution
    #         col = idx % self.resolution
    #         output[i, :] = self.img[row, col, :]
    #
    #     return output, {}


# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         image_path,
#     ):
#         super().__init__()
#         self.img = scipy.io.loadmat(image_path)['LRHSI'].astype(np.float32)
#         self.rank = 6
#         self.E = self.getE(self.img, self.rank) * 2 - 1
#         self.E = self.E.numpy()
#         self.E = np.reshape(self.E, (103, 6)).flatten()
#
#     def __len__(self):
#         return 1000
#
#     def getE(self, img, Rr):
#         ms, Ch = img.shape[0], img.shape[-1]
#
#         inters = int((Ch + 1) / (Rr + 1))
#         selected_bands = [(t + 1) * inters - 1 for t in range(Rr)]
#         band = torch.Tensor(selected_bands).type(torch.int)
#
#         img = torch.from_numpy(np.float32(img))
#         img = img.permute(2, 0, 1).unsqueeze(0)
#
#         A = torch.index_select(img, 1, band).reshape(1, Rr, -1)
#
#         t1 = torch.matmul(A, A.transpose(1, 2)) + 1e-4 * torch.eye(Rr).type(A.dtype)
#         t2 = torch.matmul(img.reshape(1, Ch, -1).cpu(), A.transpose(1, 2))
#         E = torch.matmul(t2, torch.inverse(t1))
#
#         return E
#
#     def __getitem__(self, idx):
#         return self.E, {}



# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         image_path,
#     ):
#         super().__init__()
#         self.E = scipy.io.loadmat('datasets/WhHm.mat')['W_hyper'] * 2 - 1
#         self.E = self.E.astype(np.float32)
#
#     def __len__(self):
#         return 1000
#
#     def __getitem__(self, idx):
#         i = random.randint(0, self.E.shape[1] - 1)
#         return self.E[:, i], {}
