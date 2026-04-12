import os
import h5py
import numpy as np
import scipy.signal
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import cv2

def parse_filename(filepath):
    """
    直接通过文件名前缀字符串进行绝对安全的映射，彻底解决部分类别丢失的问题。
    """
    basename = os.path.basename(filepath).replace('.mat', '')
    parts = basename.split('_')
    t_code = parts[0]  # 直接保留 'T0000', 'T0100' 等完整的字符串
    
    mapping = {
        "T0000": 0, "T0001": 1, "T0010": 2, "T0011": 3,
        "T0100": 4, "T0101": 5, "T0110": 6, "T0111": 7,
        "T1000": 8, "T1001": 9, "T1010": 10, "T1011": 11,
        "T1100": 12, "T1101": 13, "T1110": 14, "T1111": 15,
        "T10000": 16, "T10001": 17, "T10010": 18, "T10011": 19,
        "T10100": 20, "T10101": 21, "T10110": 22, "T10111": 23,
        "T11000": 24
    }
    return mapping.get(t_code, 0)

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

        # ----- 数据增强（仅在训练时开启）-----
        if self.is_train:
            # 1. 随机添加高斯噪声 (模拟宽带底噪)
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.02, mag.shape)
                mag = mag + noise
            # 2. 随机遮蔽 Cutout (模拟窄带阻塞干扰)
            if random.random() < 0.3:
                h, w = mag.shape
                mask_size = (h // 8, w // 8)
                y = random.randint(0, h - mask_size[0])
                x = random.randint(0, w - mask_size[1])
                mag[y:y+mask_size[0], x:x+mask_size[1]] = 0
        return mag

def stratified_split_files(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # 【修复】必须加上 sorted()，保证不同机器/不同次运行划分结果严格一致
    all_files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.mat')])
    
    class_to_files = defaultdict(list)
    for fpath in all_files:
        label = parse_filename(fpath)
        class_to_files[label].append(fpath)

    train_files, val_files, test_files = [], [], []

    for label, files in class_to_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_train == 0 and n >= 3: n_train = 1
        if n_val == 0 and n >= 2: n_val = 1

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

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
            raise ValueError("split must be 'train', 'val' or 'test'")

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
        except Exception:
            # 异常处理：随机采另外一个样本
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