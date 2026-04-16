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
import matplotlib.pyplot as plt
from models.swin_dual import TimeFreqDecoupledSwin

from models.resnet import ResNetBaseline
from models.alexnet import AlexNetBaseline
from models.vgg import VGG16Baseline
from torch.cuda.amp import GradScaler, autocast

from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # 25% 的概率触发 Mixup 数据增强
        use_mixup = np.random.rand() < 0.25
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
    parser.add_argument('--epochs', type=int, default=50) 
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=1e-4) 
    args = parser.parse_args()

    CHECKPOINT_DIR = os.path.abspath(config.CHECKPOINT_DIR if hasattr(config, 'CHECKPOINT_DIR') else './checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device} | AMP: True | Checkpoint Dir: {CHECKPOINT_DIR}")

    # ==========================================
    # 【核心升级】：适配高分辨率数据的动态下采样
    # ==========================================
    train_transform = T.Compose([
        T.Resize((224, 224)), # 强制将硬盘上的 1024x1024 浓缩为 Swin 支持的 224x224
        T.ToTensor(),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
        T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'), 
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)), # 验证集也必须动态缩放
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
    # 👇 添加下面这些 elif 分支
    elif args.model in ['resnet18', 'resnet50']:
        model = ResNetBaseline(num_classes=25, backbone=args.model, pretrained=True)
        save_name = args.model
    elif args.model == 'vgg16':
        model = VGG16Baseline(num_classes=25, pretrained=True)
        save_name = args.model
    elif args.model == 'alexnet':
        model = AlexNetBaseline(num_classes=25, pretrained=True)
        save_name = args.model
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")

    model = model.to(device)

    if hasattr(torch, 'compile'):
        print("🚀 检测到 PyTorch 2.0+，正在启用 torch.compile...")
        model = torch.compile(model)

    class_weights = torch.ones(25, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05) 
    
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    scaler = GradScaler()
    best_acc = 0.0
    patience = 15 
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n🚀 训练正式开始！")
    try:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | Train Acc: {(train_acc*100):.2f}% | Val Acc: {(val_acc*100):.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0 
                best_path = os.path.join(CHECKPOINT_DIR, f"{save_name}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, best_path)
                print(f"🌟 [新突破] 最优模型已更新至: {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 触发早停！最优验证集精度: {(best_acc*100):.2f}%")
                    break
                
    except KeyboardInterrupt:
        print("\n\n⚠️ 收到中止指令，紧急打包模型...")
        interrupt_path = os.path.join(CHECKPOINT_DIR, f"{save_name}_interrupted.pth")
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc if 'val_acc' in locals() else 0.0
        }, interrupt_path)
        print("您可以放心退出。\n")

    print("\n📊 正在生成训练分析曲线...")
    epochs_range = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy', marker='o', markersize=3)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o', markersize=3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plot_path = f"./results/{args.model}_{args.swin_mode}_learning_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    main()