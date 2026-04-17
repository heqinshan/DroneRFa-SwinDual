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

from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin

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
        # 核心魔法：反向传播时，梯度乘上负的 alpha，从而实现“对抗”
        output = grad_output.neg() * ctx.alpha
        return output, None

# ==========================================
# 🚀 2. 域鉴别器 (Domain Discriminator)
# ==========================================
class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=2304): # both 模式下 768 * 3 = 2304
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument('--batch_size', type=int, default=64) # 防止 OOM
    parser.add_argument('--lr', type=float, default=1e-5) 
    parser.add_argument('--source_checkpoint', type=str, default='./checkpoints/swin_dual_both_best.pth', help='热启动权重')
    args = parser.parse_args()

    CHECKPOINT_DIR = './checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    print("🌍 正在加载双域数据集 (源域: 干净信道 | 目标域: 恶劣信道)...")
    TARGET_DATA_ROOT = '/root/autodl-tmp/DroneRFa_Target_Images'

    # 开启 16 进程极致读取
    source_train_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='train', transform=transform)
    source_train_loader = DataLoader(source_train_dataset, args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    target_train_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='train', transform=transform)
    target_train_loader = DataLoader(target_train_dataset, args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    target_test_dataset = DroneRFaImageDataset(TARGET_DATA_ROOT, split='test', transform=transform)
    target_test_loader = DataLoader(target_test_dataset, args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # 初始化模型
    feature_extractor = TimeFreqDecoupledSwin(num_classes=25, pretrained=False, mode='both')
    
    # 热启动源域模型
    if os.path.exists(args.source_checkpoint):
        print(f"🔥 检测到源域神级权重，正在热启动: {args.source_checkpoint}")
        checkpoint = torch.load(args.source_checkpoint, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        uncompiled_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
        feature_extractor.load_state_dict(uncompiled_state_dict)
    else:
        print("⚠️ 未找到源域权重，将从零开始训练。")

    feature_extractor = feature_extractor.to(device)
    domain_discriminator = DomainDiscriminator(in_features=2304).to(device)

    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = nn.BCEWithLogitsLoss() 

    optimizer = optim.AdamW([
        {'params': feature_extractor.parameters(), 'lr': args.lr},
        {'params': domain_discriminator.parameters(), 'lr': args.lr * 2} 
    ], weight_decay=0.05)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    best_target_acc = 0.0
    history = {'target_acc': [], 'domain_loss': []}

    print("\n⚔️ DANN 领域对抗训练正式打响！(16进程 + 信息熵最小化 + 修复日志显示)")
    len_dataloader = min(len(source_train_loader), len(target_train_loader))

    for epoch in range(args.epochs):
        feature_extractor.train()
        domain_discriminator.train()
        
        running_c_loss, running_d_loss, running_e_loss = 0.0, 0.0, 0.0
        
        # 【修复点】：使用 zip 完美结合两个 dataloader，并且让 tqdm 正常接管
        pbar = tqdm(zip(source_train_loader, target_train_loader), total=len_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False, dynamic_ncols=True)
        
        for i, ((img_s, label_s), (img_t, _)) in enumerate(pbar):
            img_s, label_s = img_s.to(device), label_s.to(device)
            img_t = img_t.to(device)

            # 动态调节 Alpha，最高限制在 0.5
            p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
            alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.5 
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                # 1. 源域 Forward
                feat_s = feature_extractor.extract_features(img_s)
                pred_class_s = feature_extractor.fusion(feat_s)
                loss_class = class_criterion(pred_class_s, label_s)
                
                pred_domain_s = domain_discriminator(feat_s, alpha)
                label_domain_s = torch.zeros(img_s.size(0), 1).to(device) 
                loss_domain_s = domain_criterion(pred_domain_s, label_domain_s)
                
                # 2. 目标域 Forward
                feat_t = feature_extractor.extract_features(img_t)
                pred_domain_t = domain_discriminator(feat_t, alpha)
                label_domain_t = torch.ones(img_t.size(0), 1).to(device) 
                loss_domain_t = domain_criterion(pred_domain_t, label_domain_t)
                
                # 3. 目标域信息熵最小化
                pred_class_t = feature_extractor.fusion(feat_t)
                probs_t = torch.softmax(pred_class_t, dim=1)
                entropy_loss = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=1))
                
                # 4. 联合 Loss
                loss_domain = (loss_domain_s + loss_domain_t) / 2.0
                total_loss = loss_class + loss_domain

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_c_loss += loss_class.item()
            running_d_loss += loss_domain.item()
            running_e_loss += entropy_loss.item()
            
            # 【修复点】：实时更新进度条尾部的信息
            pbar.set_postfix({
                'Cls': f"{loss_class.item():.3f}", 
                'Dom': f"{loss_domain.item():.3f}", 
                'Ent': f"{entropy_loss.item():.3f}"
            })

        # 【修复点】：手动关闭进度条，防止覆盖下面的 print
        pbar.close()
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ========== 恶劣信道测试集评估 ==========
        feature_extractor.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            # 评估时显示另一个小型进度条
            for img_t, label_t in tqdm(target_test_loader, desc="Evaluating", leave=False, dynamic_ncols=True):
                img_t = img_t.to(device)
                with autocast():
                    outputs = feature_extractor(img_t)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_t.numpy())
                
        target_acc = accuracy_score(all_labels, all_preds)
        history['target_acc'].append(target_acc)
        history['domain_loss'].append(running_d_loss / len_dataloader)

        # 【修复点】：每一轮结束后的正式打印，绝对不会再被吞掉
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | Cls Loss: {(running_c_loss/len_dataloader):.4f} | Dom Loss: {(running_d_loss/len_dataloader):.4f} | Target Acc: {(target_acc*100):.2f}%")

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            best_path = os.path.join(CHECKPOINT_DIR, "swin_dann_best.pth")
            torch.save(feature_extractor.state_dict(), best_path)
            print(f"🏆 [最强鲁棒模型诞生] 目标域精度突破至: {(best_target_acc*100):.2f}%，已保存！")

    # 绘制 DANN 训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), history['target_acc'], label='Target Accuracy (Noisy Environment)', color='orange', marker='o')
    plt.title('DANN Cross-Domain Adaptation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('./results/dann_learning_curve.png', dpi=300)
    print("\n✅ DANN 训练全部结束！曲线已保存至 ./results/dann_learning_curve.png")

if __name__ == '__main__':
    main()