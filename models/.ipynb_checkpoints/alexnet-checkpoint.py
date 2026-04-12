import torch.nn as nn
import torchvision.models as models

class AlexNetBaseline(nn.Module):
    """AlexNet 基线模型，适配单通道时频图"""
    def __init__(self, num_classes=25, pretrained=False):
        super().__init__()
        # 使用 torchvision 内置 AlexNet
        self.model = models.alexnet(pretrained=pretrained)
        # 修改第一层卷积，接受单通道输入（原始是3通道）
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        # 修改分类头
        self.model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, num_classes),
)
    
    def forward(self, x):
        return self.model(x)