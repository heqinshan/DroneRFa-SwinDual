import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.autograd import Function
from torch.cuda.amp import GradScaler, autocast

# 引入您的项目配置和模型
from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin

# 开启 CUDNN 加速
torch.backends.cudnn.benchmark = True

# ==========================================
# 🚀 1. 梯度反转层 (Gradient Reversal Layer, GRL)
# ==========================================
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，梯度乘上负的 alpha，实现对抗
        output = grad_output.neg() * ctx.alpha
        return output, None

# ==========================================
# 🚀 2. 域鉴别器 (Domain Discriminator)
# ==========================================
class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=2304): # both 模式下特征维度为 768 * 3 = 2304
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 1) # 输出 1 维用于二分类 (Source=0, Target=1)
        )

    def forward(self, x, alpha):
        x = GRL.apply(x, alpha)
        return self.net(x)

# ==========================================
# 🚀 3. 主训练流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="DroneRFa DANN Target Domain Adaptation")
    parser.add_argument('--epochs', type=int, default=30, help='对抗训练轮数')
    # 默认 64 防止双域同时前向传播时导致显存溢出
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小') 
    parser.add_argument('--lr', type=float, default=5e-5, help='基础学习率')
    parser.add_argument('--source_checkpoint', type=str, 
                        default='./checkpoints/swin_dual_both_best.pth', 
                        help='源域 96.41% 的最佳权重路径')
    args = parser.parse_args()

    CHECKPOINT_DIR = './checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 高分辨率动态缩放
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    print("🌍 正在加载双域数据集 (源域: 干净信道 | 目标域: 恶劣信道)...")
    TARGET_DATA_ROOT = '/root/autodl-tmp/DroneRFa_Target_Images'

    # 源域 Dataloader (干净数据，有标签)
    source_train_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='train', transform=transform)
    source_train_loader = DataLoader(source_train_dataset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # 目标域 Dataloader (恶劣数据，仅用于特征对齐，标签被丢弃)
    target_train_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='train', transform=transform)
    target_train_loader = DataLoader(target_train_dataset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # 目标域测试集 (恶劣数据，有标签，仅用于评估真实跨域能力)
    target_test_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='test', transform=transform)
    target_test_loader = DataLoader(target_test_dataset, args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 初始化特征提取器 (Swin-Dual) 和 域分类器
    feature_extractor = TimeFreqDecoupledSwin(num_classes=config.NUM_CLASSES, pretrained=False, mode='both')
    
    # 🌟 Warm Start: 极其关键，加载源域神级权重
    if os.path.exists(args.source_checkpoint):
        print(f"🔥 检测到源域神级权重，正在执行热启动: {args.source_checkpoint}")
        checkpoint = torch.load(args.source_checkpoint, map_location='cpu')
        # 兼容处理
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        # 剔除 torch.compile 产生的 _orig_mod. 前缀
        uncompiled_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
        feature_extractor.load_state_dict(uncompiled_state_dict)
    else:
        print("⚠️ 未找到源域权重，将从零开始训练（不推荐）。")

    feature_extractor = feature_extractor.to(device)
    domain_discriminator = DomainDiscriminator(in_features=2304).to(device)

    # 损失函数
    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = nn.BCEWithLogitsLoss() 

    # 优化器
    optimizer = optim.AdamW([
        {'params': feature_extractor.parameters(), 'lr': args.lr},
        {'params': domain_discriminator.parameters(), 'lr': args.lr * 2} # 域鉴别器学习率稍大，促使对抗更激烈
    ], weight_decay=0.05)

    # 🚀 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_target_acc = 0.0
    history = {'target_acc': [], 'domain_loss': [], 'cls_loss': [], 'ent_loss': []}

    print("\n⚔️ DANN 领域对抗训练正式打响！(加入目标域信息熵最小化)")
    len_dataloader = min(len(source_train_loader), len(target_train_loader))

    for epoch in range(args.epochs):
        feature_extractor.train()
        domain_discriminator.train()
        
        running_c_loss, running_d_loss, running_e_loss = 0.0, 0.0, 0.0
        
        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)
        
        pbar = tqdm(range(len_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for i in pbar:
            try:
                img_s, label_s = next(source_iter)
                img_t, _ = next(target_iter)
            except StopIteration:
                break
                
            img_s, label_s = img_s.to(device), label_s.to(device)
            img_t = img_t.to(device)

            # 动态调节 Alpha: 从 0 平滑上升至 0.5 (上限设为0.5防止特征过度崩塌)
            p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
            alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.5 
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                # ==========================================
                # 1. 源域 Forward (分类主任务 + 域判定)
                # ==========================================
                feat_s = feature_extractor.extract_features(img_s)
                pred_class_s = feature_extractor.fusion(feat_s)
                loss_class = class_criterion(pred_class_s, label_s)
                
                pred_domain_s = domain_discriminator(feat_s, alpha)
                label_domain_s = torch.zeros(img_s.size(0), 1).to(device) # 源域标签设为 0
                loss_domain_s = domain_criterion(pred_domain_s, label_domain_s)
                
                # ==========================================
                # 2. 目标域 Forward (仅域判定 + 信息熵约束)
                # ==========================================
                feat_t = feature_extractor.extract_features(img_t)
                pred_domain_t = domain_discriminator(feat_t, alpha)
                label_domain_t = torch.ones(img_t.size(0), 1).to(device) # 目标域标签设为 1
                loss_domain_t = domain_criterion(pred_domain_t, label_domain_t)
                
                # 🚀 信息熵最小化 (Entropy Minimization): 逼迫模型在恶劣信道中给出自信的分类
                pred_class_t = feature_extractor.fusion(feat_t)
                probs_t = torch.softmax(pred_class_t, dim=1)
                entropy_loss = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=1))
                
                # ==========================================
                # 3. 联合 Loss 计算
                # ==========================================
                loss_domain = (loss_domain_s + loss_domain_t) / 2.0
                total_loss = loss_class + loss_domain + 0.1 * entropy_loss

            # 混合精度反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()