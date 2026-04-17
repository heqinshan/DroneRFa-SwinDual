import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import timm # 引入 timm 库加载标准 ViT

from config import config
from dataset import DroneRFaImageDataset

torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    CHECKPOINT_DIR = './checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    print("🌍 正在加载源域 (Source Domain) 数据集...")
    train_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    val_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='test', transform=transform)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    print("🤖 正在初始化标准 ViT (vit_base_patch16_224) 模型...")
    # 使用 timm 极速构建标准 ViT
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=25, in_chans=1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    print("\n⚔️ 标准 ViT 源域训练正式打响！(50 Epochs)")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, dynamic_ncols=True)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        pbar.close()
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True):
                imgs = imgs.to(device)
                with autocast():
                    outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        val_acc = accuracy_score(all_labels, all_preds)
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {epoch_loss:.4f} | Val Acc: {(val_acc*100):.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vit_best.pth"))
            print(f"🌟 [新突破] ViT 最优模型已保存: {(best_acc*100):.2f}%")

    print(f"\n✅ 标准 ViT 训练结束！最高精度: {(best_acc*100):.2f}%")

if __name__ == '__main__':
    main()