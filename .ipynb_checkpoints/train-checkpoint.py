import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
import matplotlib.pyplot as plt

from models.resnet import ResNetBaseline
from models.alexnet import AlexNetBaseline
from models.vgg import VGG16Baseline
from torch.cuda.amp import GradScaler, autocast

from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)

def train_one_epoch(model, ema, dataloader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # 🚀 增强 Mixup 概率到 30%，强行压制过拟合
        use_mixup = np.random.rand() < 0.30
        if use_mixup:
            lam = np.random.beta(0.2, 0.2)
            rand_index = torch.randperm(images.size()[0]).to(device)
            target_a = labels
            target_b = labels[rand_index]
            images = lam * images + (1 - lam) * images[rand_index]
            
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(images)
            if use_mixup:
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
            else:
                loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        
        if use_mixup:
            all_labels.extend(target_a.cpu().numpy() if lam > 0.5 else target_b.cpu().numpy())
        else:
            all_labels.extend(labels.cpu().numpy())
            
        all_preds.extend(preds.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(accuracy_score(all_labels, all_preds)*100):.2f}%"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val (EMA)", leave=False, dynamic_ncols=True):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast(): 
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='swin_dual')
    parser.add_argument('--swin_mode', type=str, default='both')
    parser.add_argument('--epochs', type=int, default=50) 
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--lr', type=float, default=1e-4) 
    args = parser.parse_args()

    CHECKPOINT_DIR = os.path.abspath(config.CHECKPOINT_DIR if hasattr(config, 'CHECKPOINT_DIR') else './checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 🚀 极致调优：微小面积擦除 (Micro-Erasing)
    # 面积设为 0.5% - 2%，仅仅制造微小噪点，绝不吞噬跳频线
    # ==========================================
    train_transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
        T.RandomErasing(p=0.2, scale=(0.005, 0.02), ratio=(0.3, 3.3), value=0), 
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor()
    ])

    print("正在加载极速图片数据集...")
    train_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='train', transform=train_transform)
    val_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='val', transform=val_transform)

    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, 
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4, 
        drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, 
        num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4, 
        drop_last=True 
    )

    if args.model == 'swin_dual':
        model = TimeFreqDecoupledSwin(num_classes=25, pretrained=True, mode=args.swin_mode)
        save_name = f"{args.model}_{args.swin_mode}"
    elif args.model in ['resnet18', 'resnet50']:
        model = ResNetBaseline(num_classes=25, backbone=args.model, pretrained=True)
        save_name = args.model
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")

    model = model.to(device)

    # 初始化 EMA 影子模型
    ema = ModelEMA(model, decay=0.999)

    # 🚀 恢复 Label Smoothing 为 0.1，防止模型过于自信产生过拟合
    class_weights = torch.ones(25, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05) 
    
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    scaler = GradScaler()
    best_acc = 0.0
    # 🚀 增加耐心值至 25，绝不断其后路
    patience = 25 
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n🚀 【冲刺97%】终极满血版训练开始！(重装正则化 + EMA)")
    try:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, ema, train_loader, optimizer, criterion, scaler, device)
            
            val_loss, val_acc = validate(ema.ema, val_loader, criterion, device)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | Train Acc: {(train_acc*100):.2f}% | Val Acc (EMA): {(val_acc*100):.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0 
                best_path = os.path.join(CHECKPOINT_DIR, f"{save_name}_best.pth")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ema.ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, best_path)
                print(f"🌟 [终极突破] 最优 EMA 模型已更新: {(best_acc*100):.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 触发早停！最优验证集精度: {(best_acc*100):.2f}%")
                    break
                
    except KeyboardInterrupt:
        print("\n\n⚠️ 收到中止指令...")

    print(f"\n🎉 终极训练结束！最高精度: {(best_acc*100):.2f}%")

if __name__ == '__main__':
    main()