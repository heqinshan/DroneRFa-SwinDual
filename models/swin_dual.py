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
        
        # 【防过拟合核心 1】引入 Stochastic Depth (DropPath) 和主干 Dropout
        # 这两个参数是抑制 Transformer 过拟合的“核武器”
        drop_rate = 0.3
        drop_path_rate = 0.2
        
        # 1. 时间分支
        if self.mode in ['both', 'time']:
            self.time_branch = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0, 
                in_chans=1,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate
            )
        
        # 2. 频率分支
        if self.mode in ['both', 'freq']:
            self.freq_branch = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0,
                in_chans=1,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate
            )
        
        # 3. 融合与分类头
        if self.mode == 'both':
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                batch_first=True,
                dropout=0.2
            )
            self.fusion = nn.Sequential(
                # 【架构优化】接收时域、频域、交叉注意力三个维度的特征，维度为 768 * 3
                nn.Linear(self.feature_dim * 3, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU(),
                # 【防过拟合核心 2】强化分类头的 Dropout
                nn.Dropout(0.5), 
                nn.Linear(self.feature_dim, num_classes)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU(),
                nn.Dropout(0.5), 
                nn.Linear(self.feature_dim, num_classes)
            )

    def extract_features(self, x):
        """
        【DANN 架构必备】
        专门用于提取高维特征向量。
        在训练 DANN 时，真实数据和模拟信道数据都将通过此函数输出特征，并送入域分类器进行对抗博弈。
        """
        if self.mode == 'both':
            x_time = x
            x_freq = x.transpose(2, 3)
            
            feat_time = self.time_branch.forward_features(x_time)
            feat_freq = self.freq_branch.forward_features(x_freq)
            
            if feat_time.dim() == 4:
                feat_time = feat_time.reshape(feat_time.shape[0], -1, feat_time.shape[-1])
                feat_freq = feat_freq.reshape(feat_freq.shape[0], -1, feat_freq.shape[-1])
                
            attn_out, _ = self.cross_attn(query=feat_freq, key=feat_time, value=feat_time)
            
            feat_time_pool = feat_time.mean(dim=1)
            feat_freq_pool = feat_freq.mean(dim=1)  # 【架构优化】保留纯频域全局特征
            attn_out_pool = attn_out.mean(dim=1)
            
            return torch.cat([feat_time_pool, feat_freq_pool, attn_out_pool], dim=1)
            
        elif self.mode == 'time':
            return self.time_branch(x)
            
        elif self.mode == 'freq':
            x_freq = x.transpose(2, 3)
            return self.freq_branch(x_freq)

    def forward(self, x):
        """标准有监督分类前向传播"""
        features = self.extract_features(x)
        return self.fusion(features)