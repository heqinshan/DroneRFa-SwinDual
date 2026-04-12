import numpy as np
import scipy.signal
import cv2

class STFTTransform:
    """将复数I/Q信号转换为时频图"""
    def __init__(self, nperseg=256, noverlap=128, output_size=(224, 224),
                 log_scale=True, normalize=True):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.output_size = output_size
        self.log_scale = log_scale
        self.normalize = normalize
    
    def __call__(self, signal):
        # STFT
        f, t, Zxx = scipy.signal.stft(
            signal, nperseg=self.nperseg, noverlap=self.noverlap
        )
        magnitude = np.abs(Zxx)
        
        # 对数缩放
        if self.log_scale:
            magnitude = np.log1p(magnitude)
        
        # 归一化
        if self.normalize:
            magnitude = (magnitude - magnitude.min()) / \
                        (magnitude.max() - magnitude.min() + 1e-8)
        
        # 调整尺寸
        if self.output_size:
            magnitude = cv2.resize(magnitude, self.output_size)
        
        return magnitude.astype(np.float32)


class ChannelAugmentation:
    """信道数据增强：模拟悬停抖动"""
    def __init__(self, hover_std=0.85, carrier_freq=2.4e9, sample_rate=100e6,
                 apply_prob=0.5):
        self.hover_std = hover_std
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
    
    def __call__(self, signal):
        if np.random.random() > self.apply_prob:
            return signal
        
        num_samples = len(signal)
        t = np.arange(num_samples) / self.sample_rate
        
        # 模拟速度抖动
        velocity_std = self.hover_std * (2 * np.pi * 10)
        velocity = np.random.normal(0, velocity_std, num_samples)
        displacement = np.cumsum(velocity) / self.sample_rate
        phase_rotation = 2 * np.pi * displacement / (3e8 / self.carrier_freq)
        
        # 模拟多径
        multipath = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        b, a = scipy.signal.butter(2, 20 / (self.sample_rate/2))
        multipath_fading = scipy.signal.filtfilt(b, a, multipath) * 0.2
        
        # 应用信道
        channel_coeff = np.exp(1j * phase_rotation) + multipath_fading
        signal = signal * channel_coeff
        
        return signal