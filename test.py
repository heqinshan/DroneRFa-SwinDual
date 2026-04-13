import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# 导入配置和新的图片数据集
from config import config
from dataset import DroneRFaImageDataset

from models.swin_dual import TimeFreqDecoupledSwin
from models.resnet import ResNetBaseline
from models.alexnet import AlexNetBaseline
from models.vgg import VGG16Baseline

def main():
    parser = argparse.ArgumentParser(description="DroneRFa Test Script")
    parser.add_argument('--model', type=str, default='swin_dual', choices=['swin_dual', 'resnet50', 'resnet18', 'alexnet', 'vgg16'])
    parser.add_argument('--swin_mode', type=str, default='both', choices=['both', 'time', 'freq'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=256, help='测试时的 batch_size，可以设大一点')
    args = parser.parse_args()

    os.makedirs(config.RESULT_DIR, exist_ok=True)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Testing on Device: {device}")

    # ==========================================
    # 1. 极速图片数据加载 (Test 分野)
    # ==========================================
    print("正在加载测试集图片...")
    test_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='test')
    
    # 因为是纯图片，可以拉高 batch_size 和 num_workers，瞬间测完
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=16, 
        pin_memory=True
    )

    # ==========================================
    # 2. 实例化模型
    # ==========================================
    if args.model == 'swin_dual':
        # 测试时无需预训练权重，直接从 checkpoint 加载
        model = TimeFreqDecoupledSwin(num_classes=config.NUM_CLASSES, pretrained=False, mode=args.swin_mode)
        save_name = f"{args.model}_{args.swin_mode}"
    elif args.model in ['resnet50', 'resnet18']:
        model = ResNetBaseline(num_classes=config.NUM_CLASSES, backbone=args.model, pretrained=False)
        save_name = args.model
    elif args.model == 'alexnet':
        model = AlexNetBaseline(num_classes=config.NUM_CLASSES, pretrained=False)
        save_name = args.model
    elif args.model == 'vgg16':
        model = VGG16Baseline(num_classes=config.NUM_CLASSES, pretrained=False)
        save_name = args.model

    model = model.to(device)

    # ==========================================
    # 3. 加载训练好的权重
    # ==========================================
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到权重文件: {args.checkpoint}")
        
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ==========================================
    # 4. 开始推理
    # ==========================================
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ==========================================
    # 5. 计算并打印评估指标
    # ==========================================
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    print("\n" + "="*50)
    print(f"Test Accuracy: {acc:.4f} ({(acc*100):.2f}%)")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("="*50 + "\n")

    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, digits=4)
    print(report)

    # ==========================================
    # 6. 绘制并保存极其漂亮的混淆矩阵
    # ==========================================
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14)) # 针对 25 类稍微调大了画布
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title(f'Confusion Matrix ({save_name})', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = f"{config.RESULT_DIR}/{save_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"\n混淆矩阵已高清保存至: {cm_path}")

if __name__ == '__main__':
    main()