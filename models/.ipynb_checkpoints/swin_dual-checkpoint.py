import torch
import torch.nn as nn
import timm

class TimeFreqDecoupledSwin(nn.Module):
    """
    时频解耦双分支 Swin-Transformer
    支持三种模式进行消融实验：'both' (全开), 'time' (仅时间), 'freq' (仅频率)
    """
    def __init__(self, num_classes=25, img_size=224, pretrained=True, mode='both'):
        super().__init__()
        self.mode = mode
        self.feature_dim = 768  # Swin-Tiny的特征维度
        
        # 1. 时间分支
        if self.mode in ['both', 'time']:
            self.time_branch = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0,
                in_chans=1  # 【修复】适配单通道输入
            )
        
        # 2. 频率分支
        if self.mode in ['both', 'freq']:
            self.freq_branch = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0,
                in_chans=1  # 【修复】适配单通道输入
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
            # 单分支的分类头
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
            
            # 提取序列特征 (不提前求平均)
            feat_time = self.time_branch.forward_features(x_time)  # (B, L, D)
            feat_freq = self.freq_branch.forward_features(x_freq)  # (B, L, D)
            
            # 【核心修复】激活 Cross-Attention：让频率特征作为 Query 去查询时间特征
            attn_out, _ = self.cross_attn(query=feat_freq, key=feat_time, value=feat_time)
            
            # 全局平均池化
            feat_time_pool = feat_time.mean(dim=1)
            attn_out_pool = attn_out.mean(dim=1)
            
            # 融合并分类
            feat_concat = torch.cat([feat_time_pool, attn_out_pool], dim=1)
            return self.fusion(feat_concat)
            
        elif self.mode == 'time':
            feat_time = self.time_branch.forward_features(x).mean(dim=1)
            return self.fusion(feat_time)
            
        elif self.mode == 'freq':
            x_freq = x.transpose(2, 3)
            feat_freq = self.freq_branch.forward_features(x_freq).mean(dim=1)
            return self.fusion(feat_freq)

    def forward_with_features(self, x):
        # 占位函数，避免提特征时报错
        output = self.forward(x)
        return output, torch.zeros(x.size(0), self.feature_dim).to(x.device), torch.zeros(x.size(0), self.feature_dim).to(x.device)