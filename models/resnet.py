import torch.nn as nn
import timm

class ResNetBaseline(nn.Module):
    """ResNet基线模型"""
    def __init__(self, num_classes=25, backbone='resnet50', pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=1  # 【修复】必须指定为单通道，适配 STFT 灰度图
        )
    
    def forward(self, x):
        return self.backbone(x)