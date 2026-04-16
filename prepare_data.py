import os
import shutil
import h5py
import numpy as np
import scipy.signal
import cv2
from tqdm import tqdm
from multiprocessing import Pool

# 【新增】引入你项目里的配置文件
from config import config

def parse_filename(filepath):
    """使用您原本最严谨的字典映射法提取标签"""
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

# ==========================================
# 核心高分辨率物理参数配置
# ==========================================
DATA_ROOT = config.DATA_ROOT         
OUTPUT_DIR = '/root/autodl-tmp/DroneRFa_Images'

# 物理时长 10ms，包含 1,000,000 个连续采样点
SEGMENT_LENGTH = 1000000 
# 因为每个片段变长了10倍，为了防止读取越界，每个文件的最大片段数相应缩小10倍
NUM_SEGMENTS = 150 

# ==========================================
# 自动彻底清除旧的低分辨率数据
# ==========================================
if os.path.exists(OUTPUT_DIR):
    print(f"🧹 正在删除旧的低分辨率源域数据: {OUTPUT_DIR} ...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✨ 已创建干净的新源域目录: {OUTPUT_DIR}")


def process_file(file_path):
    label = parse_filename(file_path)
    base_name = os.path.basename(file_path).replace('.mat', '')
    
    # 为每个类别建一个子文件夹
    class_dir = os.path.join(OUTPUT_DIR, str(label))
    os.makedirs(class_dir, exist_ok=True)
    
    try:
        with h5py.File(file_path, 'r') as f:
            total_len = f['RF0_I'].shape[1]
            
            for seg_idx in range(NUM_SEGMENTS):
                out_path = os.path.join(class_dir, f"{base_name}_seg{seg_idx}.png")
                if os.path.exists(out_path): 
                    continue # 如果中断了，可以断点续传
                
                start_idx = seg_idx * SEGMENT_LENGTH
                # 严格防止越界：如果剩下的数据不够 1,000,000 点，直接丢弃该文件剩余部分
                if start_idx + SEGMENT_LENGTH > total_len:
                    break 
                    
                i_data = f['RF0_I'][0, start_idx:start_idx + SEGMENT_LENGTH]
                q_data = f['RF0_Q'][0, start_idx:start_idx + SEGMENT_LENGTH]
                signal = i_data + 1j * q_data
                
                # ==========================================
                # 【核心升级】：1024 级 STFT 高清分辨率
                # ==========================================
                _, _, Zxx = scipy.signal.stft(signal, nperseg=1024, noverlap=512)
                mag = np.abs(Zxx)
                mag = np.log1p(mag)
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
                
                # 物理分辨率保存为 1024x1024
                mag = cv2.resize(mag, (1024, 1024)).astype(np.float32)
                
                # 存为 8-bit PNG 图像
                mag_img = (mag * 255).astype(np.uint8)
                cv2.imwrite(out_path, mag_img)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    all_files = [os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT) if f.endswith('.mat')]
    
    print(f"\n🚀 开始构建高分辨率 (N=1024, T=1024) 时频图数据集...")
    # 开启 16 个进程狂暴转换
    with Pool(processes=16) as pool:
        list(tqdm(pool.imap(process_file, all_files), total=len(all_files), desc="转换STFT图像"))
    print("\n✅ 全部高分辨率源域数据构建完成！您可以开始新一轮的训练了。")