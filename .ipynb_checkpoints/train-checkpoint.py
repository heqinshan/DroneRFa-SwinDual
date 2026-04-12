import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import config
from dataset import DroneRFaDataset
from transforms import STFTTransform
from models.swin_dual import TimeFreqDecoupledSwin
from models.resnet import ResNetBaseline
from utils_plot import plot_training_curves, plot_tsne
from models.alexnet import AlexNetBaseline
from models.vgg import VGG16Baseline
import torch.nn.functional as F   # 如果没有则添加


def extract_features(model, dataloader, device, max_batches=50):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for i, (images, lbls) in enumerate(dataloader):
            if i >= max_batches:
                break
            images = images.to(device)

            # ---------- 根据模型类型提取特征 ----------
            if hasattr(model, 'forward_with_features'):
                # Swin-Dual 专用
                _, feat_time, feat_freq = model.forward_with_features(images)
                feat = torch.cat([feat_time, feat_freq], dim=1)
            elif hasattr(model, 'backbone'):
                # ResNet 系列
                feat = model.backbone.forward_features(images).mean(dim=1)
            elif hasattr(model, 'model') and hasattr(model.model, 'features'):
                # AlexNet / VGG16：提取卷积特征并全局池化
                feat = model.model.features(images)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.view(feat.size(0), -1)
            else:
                # 通用回退：直接取最后一层之前的特征（简单但可能不完美）
                # 对于大多数模型，可以注册 hook 获取中间层，这里先用展平应急
                feat = images.view(images.size(0), -1)
            # -------------------------------------------

            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, use_amp=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc="Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, use_amp=True):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='swin_dual',
                        choices=['swin_dual', 'resnet50', 'resnet18', 'alexnet','vgg16'])
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = STFTTransform(nperseg=config.STFT_NPERSEG, noverlap=config.STFT_NOVERLAP,
                              output_size=(config.IMG_SIZE, config.IMG_SIZE))
    train_dataset = DroneRFaDataset(config.DATA_ROOT, transform, config.SEGMENT_LENGTH, 10, split='train', train_ratio=0.7, val_ratio=0.15)
    val_dataset = DroneRFaDataset(config.DATA_ROOT, transform, config.SEGMENT_LENGTH, 10, split='val', train_ratio=0.7, val_ratio=0.15)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)

    if args.model == 'swin_dual':
        model = TimeFreqDecoupledSwin(num_classes=config.NUM_CLASSES, pretrained=True)
        use_amp = True
    elif args.model in ['resnet50', 'resnet18']:
        model = ResNetBaseline(num_classes=config.NUM_CLASSES, backbone=args.model, pretrained=True)
        use_amp = True
    elif args.model == 'alexnet':
        model = AlexNetBaseline(num_classes=config.NUM_CLASSES, pretrained=False)
        use_amp = True
    elif args.model == 'vgg16':
        model = VGG16Baseline(num_classes=config.NUM_CLASSES, pretrained=False)
        use_amp = True
    else:
        raise ValueError(f"未知模型: {args.model}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None
    if args.model in ['alexnet', 'vgg16']:
        weight_decay = 5e-4
    else:
        weight_decay = config.WEIGHT_DECAY  # 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    start_epoch, best_acc = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'best_acc': best_acc},
                       f"{config.CHECKPOINT_DIR}/{args.model}_best.pth")

    plot_training_curves(history, f"{config.RESULT_DIR}/{args.model}_training_curves.png")
    with open(f"{config.RESULT_DIR}/{args.model}_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    checkpoint = torch.load(f"{config.CHECKPOINT_DIR}/{args.model}_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    features, labels = extract_features(model, val_loader, device)
    plot_tsne(features, labels, config.CLASS_NAMES, f"{config.RESULT_DIR}/{args.model}_tsne.png")


if __name__ == '__main__':
    main()
