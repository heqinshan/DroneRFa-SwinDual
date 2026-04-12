import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import random
import cv2

class STFTTransform:
    def __init__(self, nperseg=256, noverlap=128, output_size=(224, 224),
                 log_scale=True, normalize=True, is_train=True):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.output_size = output_size
        self.log_scale = log_scale
        self.normalize = normalize
        self.is_train = is_train

    def __call__(self, signal):
        f, t, Zxx = scipy.signal.stft(signal, nperseg=self.nperseg, noverlap=self.noverlap)
        mag = np.abs(Zxx)
        if self.log_scale:
            mag = np.log1p(mag)
        if self.normalize:
            mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        mag = cv2.resize(mag, self.output_size).astype(np.float32)

        # ----- 数据增强（仅训练时）-----
        if self.is_train:
            # 1. 随机水平翻转（时域翻转，不影响频率结构）
            if random.random() < 0.5:
                mag = np.fliplr(mag).copy()
            # 2. 随机添加高斯噪声
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.02, mag.shape)
                mag = mag + noise
            # 3. 随机遮蔽（Cutout）
            if random.random() < 0.3:
                h, w = mag.shape
                mask_size = (h // 8, w // 8)
                y = random.randint(0, h - mask_size[0])
                x = random.randint(0, w - mask_size[1])
                mag[y:y+mask_size[0], x:x+mask_size[1]] = 0
        return mag

def parse_filename(filepath):
    """
    从文件名解析标签，返回 0-24 的整数索引。
    映射关系严格按照论文表3的二进制编码。
    """
    basename = os.path.basename(filepath).replace('.mat', '')
    parts = basename.split('_')
    t_code = parts[0][1:]  # 去掉前缀 'T'
    drone_id = int(t_code, 2) if t_code else 0

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
    return mapping.get(drone_id, 0)


def stratified_split_files(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    按类别分层划分文件，返回 train/val/test 三个文件路径列表。
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. 收集所有文件并按类别分组
    all_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.mat')]
    class_to_files = defaultdict(list)
    for fpath in all_files:
        label = parse_filename(fpath)
        class_to_files[label].append(fpath)

    train_files, val_files, test_files = [], [], []

    # 2. 对每个类别进行分层划分
    for label, files in class_to_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # 确保至少有一个文件进入验证/测试集（如果数量允许）
        if n_train == 0 and n >= 3:
            n_train = 1
        if n_val == 0 and n >= 2:
            n_val = 1

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    # 打乱合并后的列表，避免训练时类别聚集
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    return train_files, val_files, test_files


class DroneRFaDataset(Dataset):
    def __init__(self, root_dir, transform=None, segment_length=100000,
                 num_segments_per_file=10, split='train', train_ratio=0.7,
                 val_ratio=0.15, return_path=False):
        self.root_dir = root_dir
        self.transform = transform
        self.segment_length = segment_length
        self.num_segments_per_file = num_segments_per_file
        self.return_path = return_path

        # 分层划分文件
        train_files, val_files, test_files = stratified_split_files(
            root_dir, train_ratio, val_ratio, 1 - train_ratio - val_ratio
        )

        if split == 'train':
            self.files = train_files
        elif split == 'val':
            self.files = val_files
        elif split == 'test':
            self.files = test_files
        else:
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split}")

        self.total_samples = len(self.files) * num_segments_per_file

        print(f"[{split.upper()}] 类别数: {len(set(parse_filename(f) for f in self.files))}, "
              f"文件数: {len(self.files)}, 总样本数: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx_in_used = idx // self.num_segments_per_file
        segment_idx = idx % self.num_segments_per_file
        file_path = self.files[file_idx_in_used]

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
            # 读取失败时随机返回另一个样本
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

    def _default_stft(self, signal):
        import scipy.signal
        import cv2
        f, t, Zxx = scipy.signal.stft(signal, nperseg=256, noverlap=128)
        mag = np.abs(Zxx)
        mag = np.log1p(mag)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        mag = cv2.resize(mag, (224, 224))
        return mag