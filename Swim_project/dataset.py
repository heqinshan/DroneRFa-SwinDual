import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def parse_filename(filepath):
    """
    从文件名解析标签，返回 0-24 的整数索引。
    映射关系严格按照论文表3的二进制编码。
    """
    basename = os.path.basename(filepath).replace('.mat', '')
    parts = basename.split('_')
    # 提取 T 字段的二进制编码部分
    t_code = parts[0][1:]  # 去掉前缀 'T'
    drone_id = int(t_code, 2) if t_code else 0
    
    # 二进制编码到连续标签 (0-24) 的映射
    # 按论文表3顺序：T0000->0, T0001->1, T0010->2, ..., T11000->24
    mapping = {
        0: 0,       # T0000 背景
        1: 1,       # T0001 Phantom 3
        2: 2,       # T0010 Phantom 4 Pro
        4: 3,       # T0011 MATRICE 200
        8: 4,       # T0100 MATRICE 100
        16: 5,      # T0101 Air 2S
        32: 6,      # T0110 Mini 3 Pro
        64: 7,      # T0111 Inspire 2
        128: 8,     # T1000 Mavic Pro
        129: 9,     # T1001 Mini 2
        256: 10,    # T1010 Mavic 3
        257: 11,    # T1011 MATRICE 300
        512: 12,    # T1100 Phantom 4 Pro RTK
        513: 13,    # T1101 MATRICE 30T
        1024: 14,   # T1110 AVATA
        2048: 15,   # T1111 DJI通信模块自组机
        4096: 16,   # T10000 MATRICE 600 Pro
        8192: 17,   # T10001 VBar
        16384: 18,  # T10010 FrSky X20
        32768: 19,  # T10011 Futaba T6IZ
        65536: 20,  # T10100 Taranis Plus
        131072: 21, # T10101 RadioLink AT9S
        262144: 22, # T10110 Futaba T14SG
        524288: 23, # T10111 云卓 T12
        1048576: 24 # T11000 云卓 T10
    }
    return mapping.get(drone_id, 0)  # 默认返回背景类


class DroneRFaDataset(Dataset):
    def __init__(self, root_dir, transform=None, segment_length=100000,
                 num_segments_per_file=10, split='train', train_ratio=0.7,
                 return_path=False):
        self.root_dir = root_dir
        self.transform = transform
        self.segment_length = segment_length
        self.num_segments_per_file = num_segments_per_file
        self.return_path = return_path
        
        self.files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.mat')])
        
        # 文件级划分（固定种子42）
        num_files = len(self.files)
        indices = np.random.RandomState(42).permutation(num_files)
        if split == 'train':
            self.file_indices = indices[:int(num_files * train_ratio)]
        elif split == 'val':
            self.file_indices = indices[int(num_files * train_ratio):int(num_files * (train_ratio + 0.15))]
        else:
            self.file_indices = indices[int(num_files * (train_ratio + 0.15)):]
        
        self.total_samples = len(self.file_indices) * num_segments_per_file
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        file_idx_in_used = idx // self.num_segments_per_file
        segment_idx = idx % self.num_segments_per_file
        file_idx = self.file_indices[file_idx_in_used]
        file_path = self.files[file_idx]
        
        max_start = 150000000 - self.segment_length
        start_idx = (segment_idx * max_start) // self.num_segments_per_file
        
        try:
            with h5py.File(file_path, 'r') as f:
                i_data = f['RF0_I'][0, start_idx:start_idx + self.segment_length]
                q_data = f['RF0_Q'][0, start_idx:start_idx + self.segment_length]
            complex_signal = i_data + 1j * q_data
            if self.transform:
                image = self.transform(complex_signal)
            else:
                image = self._default_stft(complex_signal)
            label = parse_filename(file_path)
            if self.return_path:
                return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long), file_path
            return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
    
    def _default_stft(self, signal):
        import scipy.signal, cv2
        f, t, Zxx = scipy.signal.stft(signal, nperseg=256, noverlap=128)
        mag = np.abs(Zxx)
        mag = np.log1p(mag)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        mag = cv2.resize(mag, (224, 224))
        return mag