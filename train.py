import os
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

# 引入自动混合精度 (AMP) 提速省显存
from torch.cuda.amp import GradScaler, autocast

from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin

# ==========================================
# 极致性能压榨开关 (PyTorch 黑魔法)
# ==========================================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, labels in pbar:
        # non_blocking=True 允许数据搬运和计算异步重叠
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True) # 减少显存碎片
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(accuracy_score(all_labels, all_preds)*100):.2f}%"})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
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
    parser.add_argument('--epochs', type=int, default=30) # 【设定】：默认直接跑 30 轮
    parser.add_argument('--batch_size', type=int, default=512) 
    parser.add_argument('--lr', type=float, default=3e-4) # 【设定】：配合 Warmup 提高初始峰值学习率
    args = parser.parse_args()

    CHECKPOINT_DIR = os.path.abspath(config.CHECKPOINT_DIR if hasattr(config, 'CHECKPOINT_DIR') else './checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device} | AMP: True | Checkpoint Dir: {CHECKPOINT_DIR}")

    # ==========================================
    # 数据增强：针对 STFT 频谱图的抗过拟合设计
    # ==========================================
    train_transform = T.Compose([
        T.ToTensor(),
        # 防止模型死记硬背跳频信号的绝对时间位置
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
        # value='random' 填充随机噪声，物理上等效于恶劣信道中的宽带干扰
        T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'), 
    ])
    
    val_transform = T.Compose([
        T.ToTensor()
    ])

    print("正在加载极速图片数据集...")
    train_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='train', transform=train_transform)
    val_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='val', transform=val_transform)

    # Dataloader 极致优化参数
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
    model = model.to(device)

    # ==========================================
    # 核武器级优化：PyTorch 2.0 编译加速
    # ==========================================
    if hasattr(torch, 'compile'):
        print("🚀 检测到 PyTorch 2.0+，正在启用底层计算图编译加速 (torch.compile)...")
        # 注意：使用 compile 后，第一个 Epoch 的启动会卡住 1-2 分钟进行编译，耐心等待！
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    
    # 强化 Weight Decay，从 0.05 提升至 0.1 压制过拟合
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1) 
    
    # ==========================================
    # Transformer 必备的 Warmup 调度器
    # ==========================================
    warmup_epochs = 5
    # 前 5 轮从极小学习率缓慢爬升至目标 args.lr
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    # 5 轮后接管，进行余弦退火
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    # 将两者组合
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    scaler = GradScaler()
    best_acc = 0.0
    
    # 早停机制参数
    patience = 12
    patience_counter = 0

    print("\n🚀 训练正式开始！(支持 Ctrl+C 安全紧急中断)")
    print("="*60)
    
    try:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # 步进调度器
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | Train Acc: {(train_acc*100):.2f}% | Val Acc: {(val_acc*100):.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0 # 重置早停计数器
                best_path = os.path.join(CHECKPOINT_DIR, f"{save_name}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, best_path)
                print(f"🌟 [新突破] 最优模型已更新并保存至: {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 触发早停！连续 {patience} 轮验证集无提升，训练提前结束。最优验证集精度: {(best_acc*100):.2f}%")
                    break
                
    except KeyboardInterrupt:
        print("\n\n⚠️ 收到中止指令 (Ctrl + C)！正在紧急打包当前模型...")
        interrupt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}_interrupted.pth")
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc if 'val_acc' in locals() else 0.0
        }, interrupt_path)
        print(f"✅ [紧急保存成功] 您的模型心血已安全存放在: {interrupt_path}")
        print("您可以放心退出。\n")

if __name__ == '__main__':
    main()