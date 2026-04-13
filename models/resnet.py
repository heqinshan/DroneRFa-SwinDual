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
            in_chans=1,
            drop_rate=0.5,        # 【新增】加入 50% 的全局 Dropout 防止过拟合
            drop_path_rate=0.2    # 【新增】随机深度 (Stochastic Depth) 进一步正则化
        )
    
    def forward(self, x):
        return self.backbone(x)