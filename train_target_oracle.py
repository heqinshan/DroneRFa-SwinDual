import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin

torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    CHECKPOINT_DIR = './checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    TARGET_DATA_ROOT = '/root/autodl-tmp/DroneRFa_Target_Images'
    print("🌍 正在加载恶劣信道目标域 (Target Domain) 数据集...")
    
    # 注意：这里直接使用了目标域的 train 和 test 进行监督训练
    train_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    
    val_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='test', transform=transform)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    print("🤖 正在初始化您的 Swin-Dual (Both) 模型...")
    # 从头训练 (带有局部预训练特征)，不加载源域权重
    model = TimeFreqDecoupledSwin(num_classes=25, pretrained=True, mode='both')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_acc = 0.0

    print("\n⚔️ 目标域全监督训练打响！(求取理论 Upper Bound)")
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
        
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {epoch_loss:.4f} | Target Acc: {(val_acc*100):.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 权重命名为 oracle，代表这是全知视角下的上界
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "swin_dual_target_oracle_best.pth"))
            print(f"🌟 [新突破] 理论上界已更新: {(best_acc*100):.2f}%")

    print(f"\n✅ 目标域 Upper Bound 探底结束！最高精度: {(best_acc*100):.2f}%")

if __name__ == '__main__':
    main()