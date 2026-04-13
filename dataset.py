import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import cv2
import numpy as np
from PIL import Image

def get_original_filename(image_name):
    """从图片名中提取原始 .mat 文件名 (去掉 _segX.png)"""
    return "_".join(image_name.split('_')[:-1])

class DroneRFaImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.7, val_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. 扫描所有类别文件夹并收集图片
        self.all_image_paths = []
        file_groups = defaultdict(list) # 按照原始 .mat 文件名分组
        
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            imgs = [f for f in os.listdir(cls_path) if f.endswith('.png')]
            for img in imgs:
                img_path = os.path.join(cls_path, img)
                orig_file = get_original_filename(img)
                file_groups[orig_file].append((img_path, int(cls)))

        # 2. 严格按原始文件级别进行划分，确保数据不泄露
        random.seed(seed)
        all_orig_files = sorted(list(file_groups.keys()))
        
        # 按类别对原始文件进行分层采样
        class_to_files = defaultdict(list)
        for f in all_orig_files:
            label = file_groups[f][0][1]
            class_to_files[label].append(f)
            
        train_files, val_files, test_files = [], [], []
        for label, files in class_to_files.items():
            random.shuffle(files)
            n = len(files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_files.extend(files[:n_train])
            val_files.extend(files[n_train:n_train + n_val])
            test_files.extend(files[n_train + n_val:])

        # 3. 根据 split 选定图片列表
        selected_files = {'train': train_files, 'val': val_files, 'test': test_files}[split]
        self.samples = []
        for f in selected_files:
            self.samples.extend(file_groups[f])
            
        print(f"[{split.upper()}] 原始文件数: {len(selected_files)}, 总图片张数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 【核心提速】：使用 cv2 读取单通道灰度图，速度远超 PIL
        image_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            # 必须转回 PIL 格式，因为 torchvision.transforms 需要 PIL 或 Tensor
            image = Image.fromarray(image_np)
            image = self.transform(image)
        else:
            # 验证集如果不做 transform，直接转换并归一化
            image = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0
            
        return image, torch.tensor(label, dtype=torch.long)