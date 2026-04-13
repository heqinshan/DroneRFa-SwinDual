import os
import h5py
import numpy as np
import scipy.signal
import cv2
from tqdm import tqdm
from multiprocessing import Pool


# 【新增】引入你项目里的配置文件
from config import config
def parse_filename(filepath):
    import os
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

# 【修改】直接使用 config 里的绝对路径，再也不会找不到文件了！
DATA_ROOT = config.DATA_ROOT         
OUTPUT_DIR = '/root/autodl-tmp/DroneRFa_Images'
SEGMENT_LENGTH = 100000
NUM_SEGMENTS = 1500

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                if start_idx + SEGMENT_LENGTH > total_len:
                    start_idx = total_len - SEGMENT_LENGTH
                    
                i_data = f['RF0_I'][0, start_idx:start_idx + SEGMENT_LENGTH]
                q_data = f['RF0_Q'][0, start_idx:start_idx + SEGMENT_LENGTH]
                signal = i_data + 1j * q_data
                
                # 计算 STFT
                _, _, Zxx = scipy.signal.stft(signal, nperseg=256, noverlap=128)
                mag = np.abs(Zxx)
                mag = np.log1p(mag)
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
                mag = cv2.resize(mag, (224, 224)).astype(np.float32)
                
                # 存为 8-bit PNG 图像 (极大地节省硬盘并加快读取)
                mag_img = (mag * 255).astype(np.uint8)
                cv2.imwrite(out_path, mag_img)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    all_files = [os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT) if f.endswith('.mat')]
    
    # 开启 16 个进程狂暴转换
    with Pool(processes=16) as pool:
        list(tqdm(pool.imap(process_file, all_files), total=len(all_files), desc="转换STFT图像"))
    print("全部数据离线转换完成！")