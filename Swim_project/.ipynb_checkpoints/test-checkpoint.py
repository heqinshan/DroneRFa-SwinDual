import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

from config import config
from dataset import DroneRFaDataset
from transforms import STFTTransform
from models.swin_dual import TimeFreqDecoupledSwin
from models.resnet import ResNetBaseline
from utils_plot import (plot_confusion_matrix, plot_per_class_f1, plot_roc_curves,
                        plot_precision_recall_curve, plot_gradcam)


def test(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Test")
        for images, labels in pbar:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['swin_dual', 'resnet50', 'resnet18', 'cnn'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gradcam', action='store_true', help='生成 Grad-CAM 示例图')
    args = parser.parse_args()
    
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    transform = STFTTransform(nperseg=config.STFT_NPERSEG, noverlap=config.STFT_NOVERLAP, output_size=(config.IMG_SIZE, config.IMG_SIZE))
    test_dataset = DroneRFaDataset(config.DATA_ROOT, transform, config.SEGMENT_LENGTH, 10, 'test', return_path=args.gradcam)
    test_loader = DataLoader(test_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    if args.model == 'swin_dual':
        model = TimeFreqDecoupledSwin(num_classes=config.NUM_CLASSES, pretrained=False)
    elif args.model in ['resnet50', 'resnet18']:
        model = ResNetBaseline(num_classes=config.NUM_CLASSES, backbone=args.model, pretrained=False)
    elif args.model == 'cnn':
        model = SimpleCNN(num_classes=config.NUM_CLASSES, input_channels=1)
    else:
        raise ValueError(f"未知模型: {args.model}")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if args.gradcam:
        test_dataset.return_path = True
        sample_img, sample_label, sample_path = test_dataset[0]
        sample_img = sample_img.unsqueeze(0).to(device)
        target_layer = model.time_branch.norm if args.model == 'swin_dual' else model.backbone.layer4[-1]
        plot_gradcam(model, sample_img.squeeze(0), target_layer, save_path=f"{config.RESULT_DIR}/{args.model}_gradcam.png")
        print(f"Grad-CAM 图已保存至 {config.RESULT_DIR}/{args.model}_gradcam.png")
        test_dataset.return_path = False
        test_loader = DataLoader(test_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    preds, labels, probs = test(model, test_loader, device)
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    f1_per_class = f1_score(labels, preds, average=None)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=config.CLASS_NAMES, digits=4)
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print("\nClassification Report:\n", report)
    
    results = {'accuracy': acc, 'macro_f1': f1_macro, 'weighted_f1': f1_weighted,
               'per_class_f1': f1_per_class.tolist(), 'confusion_matrix': cm.tolist()}
    with open(f"{config.RESULT_DIR}/{args.model}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_confusion_matrix(cm, config.CLASS_NAMES, f"{config.RESULT_DIR}/{args.model}_confusion.png")
    plot_per_class_f1(f1_per_class, config.CLASS_NAMES, f"{config.RESULT_DIR}/{args.model}_per_class_f1.png")
    plot_roc_curves(labels, probs, config.CLASS_NAMES, f"{config.RESULT_DIR}/{args.model}_roc.png", config.NUM_CLASSES)
    plot_precision_recall_curve(labels, probs, config.CLASS_NAMES, f"{config.RESULT_DIR}/{args.model}_pr_curve.png", config.NUM_CLASSES)
    
    print(f"\n所有结果图已保存至 {config.RESULT_DIR}/")


if __name__ == '__main__':
    main()