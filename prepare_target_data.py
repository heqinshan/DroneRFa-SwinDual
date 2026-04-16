import os
import shutil
import h5py
import numpy as np
import scipy.signal
import cv2
from tqdm import tqdm
from multiprocessing import Pool

# 引入项目配置
from config import config

# ==========================================
# 目标域核心物理参数配置 (与源域严格对齐)
# ==========================================
DATA_ROOT = config.DATA_ROOT         
TARGET_OUTPUT_DIR = '/root/autodl-tmp/DroneRFa_Target_Images'

# 物理时长 10ms，包含 1,000,000 个连续采样点
SEGMENT_LENGTH = 1000000 
# 因为每个片段变长了10倍，为了防止读取越界，每个文件的最大片段数相应缩小10倍
NUM_SEGMENTS = 150 

# ==========================================
# 自动彻底清除旧的低分辨率目标域数据
# ==========================================
if os.path.exists(TARGET_OUTPUT_DIR):
    print(f"🧹 正在删除旧的低分辨率目标域数据: {TARGET_OUTPUT_DIR} ...")
    shutil.rmtree(TARGET_OUTPUT_DIR)
os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)
print(f"✨ 已创建干净的新目标域目录: {TARGET_OUTPUT_DIR}")

def parse_filename(filepath):
    """最严谨的字典映射法提取标签"""
    basename = os.path.basename(filepath).replace('.mat', '')
    t_code = basename.split('_')[0]
    mapping = {
        "T0000": 0, "T0001": 1, "T0010": 2, "T0011": 3,
        "T0100": 4, "T0101": 5, "T0110": 6, "T0111": 7,
        "T1000": 8, "T1001": 9, "T1010": 10, "T1011": 11,
        "T1100": 12, "T1101": 13, "T1110": 14, "T1111": 15,
        "T10000": 16, "T10001": 17, "T10010": 18, "T10011": 19,
        "T10100": 20, "T10101": 21, "T10110": 22, "T10111": 23,
        "T11000": 24
    }
    return mapping.get(t_code, 0)

class ChannelAugmentation:
    """
    信道数据增强：模拟室外恶劣环境 (多普勒频移、多径衰落、低信噪比底噪)
    由于输入信号变长了(100万点)，这些物理效应将更加连贯地反映在长时频图上
    """
    def __init__(self, hover_std=0.85, carrier_freq=2.4e9, sample_rate=100e6):
        self.hover_std = hover_std
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
    
    def __call__(self, signal):
        num_samples = len(signal)
        
        # 1. 微多普勒频移 (悬停抖动)
        velocity_std = self.hover_std * (2 * np.pi * 10)
        velocity = np.random.normal(0, velocity_std, num_samples)
        displacement = np.cumsum(velocity) / self.sample_rate
        phase_rotation = 2 * np.pi * displacement / (3e8 / self.carrier_freq)
        
        # 2. 多径衰落
        multipath = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        b, a = scipy.signal.butter(2, 20 / (self.sample_rate/2))
        multipath_fading = scipy.signal.filtfilt(b, a, multipath) * 0.2
        
        channel_coeff = np.exp(1j * phase_rotation) + multipath_fading
        
        # 3. 加性高斯白噪声 (AWGN) - 掩盖微弱信号特征
        noise_power = np.mean(np.abs(signal)**2) / (10 ** (np.random.uniform(-5, 5) / 10))
        awgn = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        return signal * channel_coeff + awgn

augmenter = ChannelAugmentation()

def process_file(file_path):
    label = parse_filename(file_path)
    base_name = os.path.basename(file_path).replace('.mat', '')
    
    class_dir = os.path.join(TARGET_OUTPUT_DIR, str(label))
    os.makedirs(class_dir, exist_ok=True)
    
    try:
        with h5py.File(file_path, 'r') as f:
            total_len = f['RF0_I'].shape[1]
            
            for seg_idx in range(NUM_SEGMENTS):
                out_path = os.path.join(class_dir, f"{base_name}_seg{seg_idx}.png")
                if os.path.exists(out_path): 
                    continue # 支持断点续传
                
                start_idx = seg_idx * SEGMENT_LENGTH
                # 严格防止越界
                if start_idx + SEGMENT_LENGTH > total_len:
                    break
                    
                i_data = f['RF0_I'][0, start_idx:start_idx + SEGMENT_LENGTH]
                q_data = f['RF0_Q'][0, start_idx:start_idx + SEGMENT_LENGTH]
                signal = i_data + 1j * q_data
                
                # 在 STFT 之前，将原始 1,000,000 点 IQ 信号送入目标域信道模拟器加噪
                distorted_signal = augmenter(signal)
                
                # ==========================================
                # 【核心升级】：1024 级 STFT 高清分辨率
                # ==========================================
                _, _, Zxx = scipy.signal.stft(distorted_signal, nperseg=1024, noverlap=512)
                mag = np.abs(Zxx)
                mag = np.log1p(mag)
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
                
                # 物理分辨率保存为 1024x1024
                mag = cv2.resize(mag, (1024, 1024)).astype(np.float32)
                
                mag_img = (mag * 255).astype(np.uint8)
                cv2.imwrite(out_path, mag_img)
                
    except Exception as e:
        print(f"处理 {file_path} 失败: {e}")

if __name__ == '__main__':
    all_files = [os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT) if f.endswith('.mat')]
    
    print(f"\n🚀 开始构建高分辨率 (N=1024, T=1024) 恶劣信道目标域数据集...")
    
    # 注意：如果您此时依然在主终端运行训练任务 (train.py)，请务必将 processes=16 改为 processes=4，
    # 避免抢占过多 CPU 导致您的训练掉速。如果您是在训练完全停止的空闲状态下跑这段代码，则可以使用 16。
    with Pool(processes=4) as pool:
        list(tqdm(pool.imap(process_file, all_files), total=len(all_files), desc="构建目标域(加噪)图像"))
    print("\n✅ 高分辨率目标域数据全部构建完成！")