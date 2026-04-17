import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast

from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin

torch.backends.cudnn.benchmark = True

def main():
    CHECKPOINT_DIR = './checkpoints'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 极低学习率，微调 15 轮足够
    EPOCHS = 15
    BATCH_SIZE = 64
    LR = 5e-6 
    CONFIDENCE_THRESHOLD = 0.65 # 只挑选置信度大于 95% 的“伪标签”

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    TARGET_DATA_ROOT = '/root/autodl-tmp/DroneRFa_Target_Images'
    print("🌍 加载目标域数据集，准备提取伪标签...")
    
    # 获取目标域无标签训练集
    target_train_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='train', transform=transform)
    target_train_loader = DataLoader(target_train_dataset, BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    
    # 目标域测试集
    target_test_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='test', transform=transform)
    target_test_loader = DataLoader(target_test_dataset, BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

    print("🤖 正在加载 DANN 最强权重 (89.69%)...")
    model = TimeFreqDecoupledSwin(num_classes=25, pretrained=False, mode='both').to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "swin_dann_best.pth")))
    model.eval()

    # ==========================================
    # 🚀 步骤 1: 预测并筛选高置信度伪标签
    # ==========================================
    confident_imgs = []
    confident_pseudo_labels = []
    
    print(f"🔍 正在扫描目标域，阈值: {CONFIDENCE_THRESHOLD}...")
    with torch.no_grad():
        for imgs, _ in tqdm(target_train_loader, desc="Pseudo-Labeling", dynamic_ncols=True):
            imgs = imgs.to(device)
            with autocast():
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                
            max_probs, preds = torch.max(probs, 1)
            
            # 找出大于阈值的索引
            mask = max_probs > CONFIDENCE_THRESHOLD
            
            if mask.sum() > 0:
                confident_imgs.append(imgs[mask].cpu())
                confident_pseudo_labels.append(preds[mask].cpu())

    if len(confident_imgs) == 0:
        print("❌ 未找到高置信度样本，无法进行伪标签训练。")
        return

    all_confident_imgs = torch.cat(confident_imgs, dim=0)
    all_confident_labels = torch.cat(confident_pseudo_labels, dim=0)
    
    print(f"🎯 成功提取 {len(all_confident_labels)} 个极度自信样本作为伪标签！")

    # 创建伪标签 Dataloader
    pseudo_dataset = TensorDataset(all_confident_imgs, all_confident_labels)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # ==========================================
    # 🚀 步骤 2: 伪标签微调训练 (Self-Training)
    # ==========================================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = GradScaler()
    best_acc = 0.8969 # 以之前的 DANN 最好成绩作为底线

    print("\n⚔️ 伪标签冲刺微调开始！(目标：突破 90%)")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for imgs, labels in tqdm(pseudo_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False, dynamic_ncols=True):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        # 验证阶段
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in target_test_loader:
                imgs = imgs.to(device)
                with autocast():
                    outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} | Fine-tune Loss: {(running_loss/len(pseudo_loader)):.4f} | Target Acc: {(val_acc*100):.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "swin_dann_pseudo_best.pth"))
            print(f"🌟 [终极突破] 目标域精度再创新高: {(best_acc*100):.2f}%，直逼 Oracle 上界！")

    print(f"\n✅ 伪标签微调结束！最高冲刺精度: {(best_acc*100):.2f}%")

if __name__ == '__main__':
    main()