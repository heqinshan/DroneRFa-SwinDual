import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch
import torch.nn.functional as F
import cv2

# 设置统一的学术风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

COLORS = sns.color_palette("husl", 25)


def plot_training_curves(history, save_path):
    """训练曲线：Loss & Accuracy"""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', lw=2, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', lw=2, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', lw=2, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', lw=2, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, normalize=True):
    """归一化混淆矩阵热力图"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = None
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={"shrink": 0.8})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_f1(f1_scores, class_names, save_path):
    """各类别 F1-score 柱状图（降序排列）"""
    sorted_idx = np.argsort(f1_scores)[::-1]
    sorted_names = [class_names[i] for i in sorted_idx]
    sorted_f1 = [f1_scores[i] for i in sorted_idx]
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(sorted_f1)), sorted_f1, color=COLORS)
    plt.xticks(range(len(sorted_f1)), sorted_names, rotation=90)
    plt.ylabel('F1-Score')
    plt.xlabel('Class')
    plt.title('Per-Class F1-Score (Sorted)')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_tsne(features, labels, class_names, save_path, max_samples=2000):
    """t-SNE 特征分布可视化"""
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels, cmap='tab20', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.title('t-SNE Visualization of Learned Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(y_true, y_score, class_names, save_path, n_classes=25):
    """多分类 ROC 曲线（宏平均 + 各类别）"""
    y_true_onehot = np.eye(n_classes)[y_true]
    
    plt.figure(figsize=(10, 8))
    # 宏平均
    fpr, tpr, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', lw=2, label=f'Macro-average (AUC = {roc_auc:.3f})')
    
    # 各类别
    for i in range(n_classes):
        if np.sum(y_true == i) > 0:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, alpha=0.7,
                    label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curve(y_true, y_score, class_names, save_path, n_classes=25):
    """精确率-召回率曲线"""
    y_true_onehot = np.eye(n_classes)[y_true]
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        if np.sum(y_true == i) > 0:
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
            plt.plot(recall, precision, lw=1, alpha=0.7, label=class_names[i])
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gradcam(model, image_tensor, target_layer, class_idx=None, save_path=None):
    """
    Grad-CAM 热力图生成
    适用于 Swin-Transformer 的特定层
    """
    model.eval()
    # 确保输入张量需要梯度
    image_tensor = image_tensor.unsqueeze(0).cuda()
    image_tensor.requires_grad = True

    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    if class_idx is None:
        class_idx = output.argmax().item()

    model.zero_grad()
    output[0, class_idx].backward()

    feature_map = features[0].detach()
    grad = gradients[0].detach()

    # 处理 Swin Transformer 的序列输出
    if len(feature_map.shape) == 3:  # (1, L, D)
        L = feature_map.shape[1]
        H = W = int(np.sqrt(L))
        feature_map = feature_map.view(1, H, W, -1).permute(0, 3, 1, 2)
        grad = grad.view(1, H, W, -1).permute(0, 3, 1, 2)

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_map).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = F.interpolate(cam, size=(224, 224), mode='bilinear')
    cam = cam.squeeze().cpu().numpy()

    # ---------- 修复图像形状处理 ----------
    img = image_tensor.detach().squeeze().cpu().numpy()
    if img.ndim == 2:  # (H, W)
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3:
        if img.shape[0] == 1:  # (1, H, W)
            img = np.stack([img[0]]*3, axis=-1)
        elif img.shape[0] == 3:  # (3, H, W)
            img = img.transpose(1, 2, 0)
        else:
            img = img.transpose(1, 2, 0)  # fallback
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img * 255 + 0.5 * cam_heatmap
    overlay = overlay.astype(np.uint8)

    if save_path:
        plt.imsave(save_path, overlay)

    handle_forward.remove()
    handle_backward.remove()
    return overlay


def plot_model_comparison(compare_dict, save_path):
    """
    绘制多模型性能对比柱状图
    """
    models = list(compare_dict.keys())
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    
    # 提取数据
    accs = [compare_dict[m]['accuracy'] for m in models]
    macro_f1s = [compare_dict[m]['macro_f1'] for m in models]
    weighted_f1s = [compare_dict[m]['weighted_f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, accs, width, label='Accuracy', color='skyblue')
    rects2 = ax.bar(x, macro_f1s, width, label='Macro F1', color='lightcoral')
    rects3 = ax.bar(x + width, weighted_f1s, width, label='Weighted F1', color='lightgreen')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # ==========================================
    # 【核心修复 1】：把 Y 轴上限提高到 1.08，给长数字留出“天空”
    # ==========================================
    ax.set_ylim(0.8, 1.08) 
    
    # ==========================================
    # 【核心修复 2】：图例依然放在左上角，但去掉了边框，更清爽
    # ==========================================
    ax.legend(loc='upper left', frameon=False) 
    
    # 自动在柱子上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 稍微再把数字往上抬高 4 个像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, rotation=90)
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_radar_chart(metrics_dict, class_names, save_path, top_k=10):
    """雷达图展示模型在各类别上的综合性能"""
    from math import pi
    
    categories = class_names[:top_k]
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    for model_name, values in metrics_dict.items():
        vals = values[:top_k]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, vals, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Per-Class F1-Score Radar Chart')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()