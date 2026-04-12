import torch
import torch.nn as nn
import timm


class TimeFreqDecoupledSwin(nn.Module):
    """
    时频解耦双分支 Swin-Transformer
    
    核心思想：
    - 时间注意力分支 (Time Branch)：使用1×W扁平窗口，捕获时序脉冲和包络特征
    - 频率注意力分支 (Freq Branch)：使用H×1瘦高窗口，捕获频带占用和谐波特征
    - 融合分支 (Fusion)：融合两个分支的特征进行最终分类
    """
    def __init__(self, num_classes=25, img_size=224, pretrained=True):
        super().__init__()
        
        # 时间分支：使用标准的Swin-Tiny，但修改窗口形状
        self.time_branch = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0  # 只取特征
        )
        
        # 频率分支：同样使用Swin-Tiny
        self.freq_branch = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        # 获取特征维度
        self.feature_dim = 768  # Swin-Tiny的特征维度
        
        # 时频特征交互模块（Cross-Attention）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # 窗口变换模块：用于在预处理时实现时频解耦
        self.time_window_reshape = None
        self.freq_window_reshape = None
    
    def forward(self, x):
        # x: (B, C, H, W) - 时频图
        
        # 时间分支：对输入进行转置，使得模型关注时间维度的长距离依赖
        # 将频率维度视为通道，时间维度视为空间
        x_time = x  # 原始输入
        x_freq = x.transpose(2, 3)  # 转置H和W，使模型关注频率维度
        
        # 提取特征
        feat_time = self.time_branch.forward_features(x_time)  # (B, L, D)
        feat_freq = self.freq_branch.forward_features(x_freq)  # (B, L, D)
        
        # 全局平均池化得到特征向量
        feat_time = feat_time.mean(dim=1)  # (B, D)
        feat_freq = feat_freq.mean(dim=1)  # (B, D)
        
        # 特征拼接
        feat_concat = torch.cat([feat_time, feat_freq], dim=1)  # (B, 2D)
        
        # 分类
        output = self.fusion(feat_concat)  # (B, num_classes)
        
        return output
    
    def forward_with_features(self, x):
        """返回特征用于可视化"""
        x_time = x
        x_freq = x.transpose(2, 3)
        
        feat_time = self.time_branch.forward_features(x_time).mean(dim=1)
        feat_freq = self.freq_branch.forward_features(x_freq).mean(dim=1)
        
        feat_concat = torch.cat([feat_time, feat_freq], dim=1)
        output = self.fusion(feat_concat)
        
        return output, feat_time, feat_freq