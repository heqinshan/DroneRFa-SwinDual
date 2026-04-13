import torch
import torch.nn as nn
import timm

class TimeFreqDecoupledSwin(nn.Module):
    """
    时频解耦双分支 Swin-Transformer
    支持三种模式进行消融实验：'both', 'time', 'freq'
    """
    def __init__(self, num_classes=25, img_size=224, pretrained=True, mode='both'):
        super().__init__()
        self.mode = mode
        self.feature_dim = 768  # Swin-Tiny的特征维度
        
        # 1. 时间分支 (仅在 both 或 time 模式下加载)
        if self.mode in ['both', 'time']:
            self.time_branch = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0, # 设为0代表获取全局池化后的特征
                in_chans=1 
            )
        
        # 2. 频率分支 (仅在 both 或 freq 模式下加载)
        if self.mode in ['both', 'freq']:
            self.freq_branch = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0,
                in_chans=1
            )
        
        # 3. 融合与分类头
        if self.mode == 'both':
            # 时频特征交互模块 (Cross-Attention)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                batch_first=True,
                dropout=0.1
            )
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.feature_dim, num_classes)
            )
        else:
            # 单分支模式的分类头
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.feature_dim, num_classes)
            )
    
    def forward(self, x):
        if self.mode == 'both':
            x_time = x
            x_freq = x.transpose(2, 3)
            
            # 提取序列特征 (为了做交叉注意力，必须用 forward_features)
            feat_time = self.time_branch.forward_features(x_time)
            feat_freq = self.freq_branch.forward_features(x_freq)
            
            # 【核心修复】：兼容 timm 不同版本返回的 4D 张量 (B, H, W, C)
            # 强制展平高宽维度，转化为标准的 3D 序列 (B, L, C)，如 (32, 49, 768)
            if feat_time.dim() == 4:
                feat_time = feat_time.reshape(feat_time.shape[0], -1, feat_time.shape[-1])
                feat_freq = feat_freq.reshape(feat_freq.shape[0], -1, feat_freq.shape[-1])
                
            attn_out, _ = self.cross_attn(query=feat_freq, key=feat_time, value=feat_time)
            
            # 全局平均池化
            feat_time_pool = feat_time.mean(dim=1)
            attn_out_pool = attn_out.mean(dim=1)
            
            feat_concat = torch.cat([feat_time_pool, attn_out_pool], dim=1)
            return self.fusion(feat_concat)
            
        elif self.mode == 'time':
            # 【核心修复】：单分支模式直接调用整个分支网络
            # timm会自动处理好所有的全局平均池化，安全返回 (B, 768)
            feat_time = self.time_branch(x)
            return self.fusion(feat_time)
            
        elif self.mode == 'freq':
            # 【核心修复】：单分支模式直接调用整个分支网络
            x_freq = x.transpose(2, 3)
            feat_freq = self.freq_branch(x_freq)
            return self.fusion(feat_freq)

    def forward_with_features(self, x):
        """用于 T-SNE 特征提取的占位回退函数"""
        output = self.forward(x)
        return output, torch.zeros(x.size(0), self.feature_dim).to(x.device), torch.zeros(x.size(0), self.feature_dim).to(x.device)