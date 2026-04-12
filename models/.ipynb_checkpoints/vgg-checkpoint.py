import torch.nn as nn
import torchvision.models as models

class VGG16Baseline(nn.Module):
    """VGG16 基线模型，适配单通道时频图"""
    def __init__(self, num_classes=25, pretrained=False):
        super().__init__()
        self.model = models.vgg16(pretrained=pretrained)
        # 修改第一层卷积，接受单通道输入
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # 修改分类头
        self.model.classifier[6] = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        return self.model(x)