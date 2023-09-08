import os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from torch.utils.data import Dataset

class ScansDataLoader(Dataset):
    def __init__(self, dir_path: str, scans: List[str], lidar: str = 'vlp-16') -> None:
        super().__init__()

        assert lidar in {'vlp-16', 'hdl-32', 'hdl-64'}, 'lidar type should be \'vlp-16\' or \'hdl-32\''

        self.frames = []
        self.scan_id = []

        lidar_params = {
            'vlp-16': {
                'num_beams': 16,
                'fov_up': 16.8,
                'fov_down': -16.8,
                'window_size': 128,
            },
            'hdl-32': {
                'num_beams': 32,
                'fov_up': 30,
                'fov_down': -10,
                'window_size': 64,
            },
            'hdl-64': {
                'num_beams': 64,
                'fov_up': 2,
                'fov_down': -24,
                'window_size': 32,
            },
        }
        lidar_param = lidar_params[lidar]

        self.num_slices = 1024
        self.window_size = lidar_param['window_size']
        self.num_windows = self.num_slices // self.window_size

        for scan in tqdm(scans, total=len(scans)):
            data = np.load(os.path.join(dir_path, scan))
            frame = self.lidar_to_image(data, lidar_param, self.num_slices)
            self.frames.append(frame)
            self.scan_id.append(scan[:-4])

        scan_idxs = [idx // self.num_windows for idx in range(len(scans) * self.num_windows)]
        self.scan_idxs = np.array(scan_idxs)
        print(f"Total samples {len(self.scan_idxs)}.")


    def lidar_to_image(self, data: np.ndarray, lidar_param: dict, num_slices: int) -> np.ndarray:
        """
        Convert LiDAR points to an image.

        Args:
        data: A numpy array of shape (N, 4), where N is the number of points and each row
            contains (x, y, z, s) coordinates, where 's' is the stability score.
        lidar_param->num_beams: Number of beams in the LiDAR sensor.
        lidar_param->fov_up: Field of view angle in degrees in the upward direction.
        lidar_param->fov_down: Field of view angle in degrees in the downward direction.

        num_slices: Number of slices (azimuth angle) in the range image.

        Returns:
        range_image: A numpy array of shape (ibrahim@10.5.37.139:/home/ibrahim/neptune/pn2-reg/data/., num_slices) representing the range image.
        """

        data = np.unique(data, axis=0)
        data = data[data[:, 3] != -1]

        x, y, z, s = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        theta = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
        phi = np.arctan2(y, x) * 180 / np.pi

        fov_total = lidar_param['fov_up'] - lidar_param['fov_down']
        theta_res = fov_total / (lidar_param['num_beams'] - 1)
        phi_res = 360 / num_slices

        theta_idx = np.floor((theta - lidar_param['fov_down']) / theta_res).astype(np.int32)
        phi_idx = np.floor(phi / phi_res).astype(np.int32)

        num_channels = 4
        range_image = np.zeros((lidar_param['num_beams'], num_slices, num_channels), dtype=np.float32)

        # adjust the indices to ensure they are within bounds
        theta_idx[theta_idx >= lidar_param['num_beams']] = lidar_param['num_beams'] - 1

        range_image[theta_idx, phi_idx, 0] = x
        range_image[theta_idx, phi_idx, 1] = y
        range_image[theta_idx, phi_idx, 2] = z
        range_image[theta_idx, phi_idx, 3] = s

        return range_image


    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        scan_idx = self.scan_idxs[idx]
        frame = self.frames[scan_idx]
        window_num = idx % self.num_windows

        w_s = window_num * self.window_size
        w_e = w_s + self.window_size

        frame = frame[:, w_s:w_e, :].reshape(-1, frame.shape[-1])

        points = frame[:, :3]
        labels = frame[:, 3]

        return points, labels, f"{self.scan_id[scan_idx]}_{window_num}"

    def __len__(self) -> int:
        return len(self.scan_idxs)
