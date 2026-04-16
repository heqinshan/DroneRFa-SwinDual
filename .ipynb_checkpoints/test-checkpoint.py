import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import random
import json
import torchvision.transforms as T  # 【修复 1】：引入 transforms

from config import config
from dataset import DroneRFaImageDataset
from models.swin_dual import TimeFreqDecoupledSwin
from models.resnet import ResNetBaseline
from models.alexnet import AlexNetBaseline
from models.vgg import VGG16Baseline

# 导入所有可视化函数
from utils_plot import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_tsne,
    plot_roc_curves,
    plot_precision_recall_curve,
    plot_gradcam,
    plot_training_curves,
    plot_model_comparison,
    plot_radar_chart
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def extract_features(model, dataloader, device, max_samples=5000):
    """从数据加载器中提取特征和标签，最多采集 max_samples 个样本"""
    if hasattr(model, '_orig_mod'):
        raw_model = model._orig_mod
    else:
        raw_model = model

    raw_model.eval()
    features, labels_list = [], []
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting features for t-SNE"):
            images = images.to(device)

            if hasattr(raw_model, 'time_branch') and hasattr(raw_model, 'freq_branch'):
                feat_time = raw_model.time_branch.forward_features(images)
                if feat_time.dim() == 3:
                    feat_time = feat_time.mean(dim=1)
                feat_time = feat_time.view(feat_time.size(0), -1)

                images_transposed = images.transpose(2, 3)
                feat_freq = raw_model.freq_branch.forward_features(images_transposed)
                if feat_freq.dim() == 3:
                    feat_freq = feat_freq.mean(dim=1)
                feat_freq = feat_freq.view(feat_freq.size(0), -1)

                feat = torch.cat([feat_time, feat_freq], dim=1)
            elif hasattr(raw_model, 'backbone'):
                feat = raw_model.backbone.forward_features(images)
                if feat.dim() == 3:
                    feat = feat.mean(dim=1)
                feat = feat.view(feat.size(0), -1)
            elif hasattr(raw_model, 'model') and hasattr(raw_model.model, 'features'):
                feat = raw_model.model.features(images)
                feat = F.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.view(feat.size(0), -1)
            else:
                raise AttributeError(f"无法从 {type(raw_model).__name__} 提取特征，请实现专用分支。")

            features.append(feat.cpu().numpy())
            labels_list.extend(lbls.numpy())

            if len(labels_list) >= max_samples:
                break

    features = np.concatenate(features, axis=0)[:max_samples]
    labels_arr = np.array(labels_list[:max_samples])
    return features, labels_arr


def main():
    parser = argparse.ArgumentParser(description="DroneRFa Comprehensive Test Script")
    parser.add_argument('--model', type=str, default='swin_dual',
                        choices=['swin_dual', 'resnet50', 'resnet18', 'alexnet', 'vgg16'])
    parser.add_argument('--swin_mode', type=str, default='both', choices=['both', 'time', 'freq'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tsne', action='store_true', help='生成 t-SNE 图')
    parser.add_argument('--tsne_samples', type=int, default=5000)
    parser.add_argument('--gradcam', action='store_true', help='生成 Grad-CAM 热力图')
    parser.add_argument('--history', type=str, help='训练历史 JSON 文件路径（生成训练曲线）')
    parser.add_argument('--compare', nargs='+', help='多个结果 JSON 文件路径，用于模型对比图')
    args = parser.parse_args()

    os.makedirs(config.RESULT_DIR, exist_ok=True)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Testing on Device: {device}")

    # ==========================================
    # 【修复 2】：为测试集添加动态 Resize
    # ==========================================
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    print("正在加载测试集图片...")
    test_dataset = DroneRFaImageDataset(config.IMAGE_DATA_ROOT, split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=16, pin_memory=True)

    if args.tsne:
        num_samples = min(args.tsne_samples, len(test_dataset))
        indices = random.sample(range(len(test_dataset)), num_samples)
        tsne_dataset = Subset(test_dataset, indices)
        tsne_loader = DataLoader(tsne_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)
        print(f"已从测试集采样 {num_samples} 张图片用于 t-SNE。")

    # 实例化模型
    if args.model == 'swin_dual':
        model = TimeFreqDecoupledSwin(num_classes=config.NUM_CLASSES,
                                      pretrained=False, mode=args.swin_mode)
        save_name = f"{args.model}_{args.swin_mode}"
    elif args.model in ['resnet50', 'resnet18']:
        model = ResNetBaseline(num_classes=config.NUM_CLASSES,
                               backbone=args.model, pretrained=False)
        save_name = args.model
    elif args.model == 'alexnet':
        model = AlexNetBaseline(num_classes=config.NUM_CLASSES, pretrained=False)
        save_name = args.model
    elif args.model == 'vgg16':
        model = VGG16Baseline(num_classes=config.NUM_CLASSES, pretrained=False)
        save_name = args.model

    model = model.to(device)

    # 加载权重
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到权重文件: {args.checkpoint}")
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 兼容是否保存了完整的 dict 还是单独存了 state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 去除 torch.compile 产生的 _orig_mod. 前缀
    uncompiled_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[10:] if k.startswith("_orig_mod.") else k
        uncompiled_state_dict[new_key] = v
    model.load_state_dict(uncompiled_state_dict)
    model.eval()

    # ========== t-SNE ==========
    if args.tsne:
        print("正在提取特征用于 t-SNE...")
        features, tsne_labels = extract_features(model, tsne_loader, device, max_samples=args.tsne_samples)
        tsne_path = os.path.join(config.RESULT_DIR, f"{save_name}_tsne.png")
        plot_tsne(features, tsne_labels, config.CLASS_NAMES, tsne_path)
        print(f"t-SNE 图已保存至: {tsne_path}")

    # ========== Grad-CAM ==========
    if args.gradcam:
        print("正在生成 Grad-CAM 热力图...")
        sample_img, sample_label = test_dataset[0]
        sample_img = sample_img.unsqueeze(0).to(device)
        if hasattr(model, '_orig_mod'):
            raw_model = model._orig_mod
        else:
            raw_model = model
        if hasattr(raw_model, 'time_branch'):
            target_layer = raw_model.time_branch.norm
        elif hasattr(raw_model, 'backbone'):
            target_layer = raw_model.backbone.layer4[-1]
        elif hasattr(raw_model, 'model'):
            target_layer = raw_model.model.features[-1]
        else:
            target_layer = None
            print("警告：无法自动确定 Grad-CAM 目标层，跳过。")
        if target_layer is not None:
            gradcam_path = os.path.join(config.RESULT_DIR, f"{save_name}_gradcam.png")
            # 传入模型和原始张量
            plot_gradcam(model, sample_img.squeeze(0), target_layer, save_path=gradcam_path)
            print(f"Grad-CAM 图已保存至: {gradcam_path}")

    # ========== 常规测试 ==========
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 指标
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*50)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print("="*50 + "\n")
    print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, digits=4))

    # 保存结果 JSON
    results = {
        'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1.tolist(), 'confusion_matrix': cm.tolist()
    }
    results_path = os.path.join(config.RESULT_DIR, f"{save_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制所有图表
    plot_confusion_matrix(cm, config.CLASS_NAMES,
                          os.path.join(config.RESULT_DIR, f"{save_name}_confusion.png"))
    plot_per_class_f1(per_class_f1, config.CLASS_NAMES,
                      os.path.join(config.RESULT_DIR, f"{save_name}_per_class_f1.png"))
    plot_roc_curves(all_labels, all_probs, config.CLASS_NAMES,
                    os.path.join(config.RESULT_DIR, f"{save_name}_roc.png"), config.NUM_CLASSES)
    plot_precision_recall_curve(all_labels, all_probs, config.CLASS_NAMES,
                                os.path.join(config.RESULT_DIR, f"{save_name}_pr_curve.png"), config.NUM_CLASSES)

    # 训练曲线（如果提供了 history JSON）
    if args.history and os.path.exists(args.history):
        with open(args.history, 'r') as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(config.RESULT_DIR, f"{save_name}_training_curves.png"))

    # 模型对比（如果提供了多个结果 JSON）
    if args.compare:
        compare_dict = {}
        for res_file in args.compare:
            name = os.path.basename(res_file).replace('_results.json', '')
            with open(res_file, 'r') as f:
                compare_dict[name] = json.load(f)
        plot_model_comparison(compare_dict, os.path.join(config.RESULT_DIR, "model_comparison.png"))
        radar_data = {name: data['per_class_f1'] for name, data in compare_dict.items()}
        plot_radar_chart(radar_data, config.CLASS_NAMES,
                         os.path.join(config.RESULT_DIR, "model_radar.png"), top_k=10)

    print(f"\n所有结果已保存至 {config.RESULT_DIR}/")


if __name__ == '__main__':
    main()