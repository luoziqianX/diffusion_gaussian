import torch
import numpy as np
import os
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement


class GaussianDataset(Dataset):
    def __init__(self, dataset_dir: str, min_points: int = 8):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.min_points = min_points
        self.filenames = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir) if
                          filename.endswith('.ply')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        plydata = PlyData.read(filename)
        xyz = np.stack((np.asarray(plydata.elements[0]['x']), np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])), axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        xyz = torch.tensor(xyz).float()
        opacities = torch.tensor(opacities).float()
        scales = torch.tensor(scales).float()
        rots = torch.tensor(rots).float()
        features_dc = torch.tensor(features_dc).float()

        gaussians = torch.cat([xyz, opacities.unsqueeze(1), scales, rots, features_dc], dim=1)

        gaussians = gaussians[:gaussians.shape[0] - gaussians.shape[0] % self.min_points, :]
        gaussians = gaussians.permute(1, 0)

        return gaussians
