import os
import h5py
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# ==========================================
# 🚀 1. 论文级可视化配置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ==========================================
# 🚀 2. 物理参数配置
# ==========================================
DATA_ROOT = "/root/autodl-tmp/DroneRFa" # 替换为您的真实 .mat 文件所在目录
OUTPUT_DIR = "./results/Paper_Figures/T0111_Matched"

FS = 100e6               # 采样率 100 MHz
DURATION = 0.1           # 截取时长 0.1 s
NUM_SAMPLES = int(FS * DURATION) # 对应 10,000,000 个采样点

NPERSEG = 2048           # FFT 窗长
WINDOW = 'hamming'       # 汉明窗

# ==========================================
# 🚀 3. 极度恶劣的信道模拟器 (更强噪声、衰落与频移)
# ==========================================
def apply_extreme_channel(signal, fs=100e6):
    num_samples = len(signal)
    
    # 1. 狂暴微多普勒 (大风环境剧烈抖动)
    velocity_std = 2.0 * (2 * np.pi * 15) # 参数加大
    velocity = np.random.normal(0, velocity_std, num_samples)
    displacement = np.cumsum(velocity) / fs
    phase_rotation = 2 * np.pi * displacement / (3e8 / 2.4e9)
    
    # 2. 深度多径衰落 (城市峡谷严重遮挡)
    multipath = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    b, a = scipy.signal.butter(2, 50 / (fs/2)) 
    multipath_fading = scipy.signal.filtfilt(b, a, multipath) * 0.6 # 衰落幅度加大
    
    channel_coeff = np.exp(1j * phase_rotation) + multipath_fading
    
    # 3. 极限低信噪比 (极强底噪，SNR = -8 dB)
    target_snr_db = -8
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    awgn = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    
    return signal * channel_coeff + awgn

# ==========================================
# 🚀 4. STFT 计算核心 (复信号完整频谱)
# ==========================================
def compute_stft_db(sig, fs=100e6, nperseg=2048, window='hamming'):
    # 计算 STFT，return_onesided=False 因为是复信号 (IQ)，包含正负频率
    f, t, Zxx = scipy.signal.stft(sig, fs=fs, window=window, nperseg=nperseg, 
                                  noverlap=nperseg//2, return_onesided=False)
    
    # 将 0Hz 移到中心
    Zxx = np.fft.fftshift(Zxx, axes=0)
    f = np.fft.fftshift(f)
    
    # 转换为 dB (Magnitude -> dB)
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(mag + 1e-12) # 加极小值防止 log(0)
    
    # 将频率转为 MHz，时间为 s
    f_mhz = f / 1e6
    return f_mhz, t, mag_db

# ==========================================
# 🚀 5. 绘图函数
# ==========================================
def plot_and_save(f_mhz, t_s, mag_db, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用 imshow 进行极速高清渲染
    im = ax.imshow(mag_db, aspect='auto', origin='lower', cmap='jet',
                   extent=[t_s[0], t_s[-1], f_mhz[0], f_mhz[-1]],
                   vmin=np.percentile(mag_db, 5), vmax=np.percentile(mag_db, 99.5)) # 截断极端值，让色彩更通透
    
    # 坐标轴设置
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Frequency (MHz)', fontweight='bold')
    
    # 右侧 Colorbar 设置 (dB)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"✅ 已保存: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 在 DATA_ROOT 中寻找一个 T0111 开头的文件
    target_file = None
    if os.path.exists(DATA_ROOT):
        for f_name in os.listdir(DATA_ROOT):
            if f_name.startswith('T0111') and f_name.endswith('.mat'):
                target_file = os.path.join(DATA_ROOT, f_name)
                break
    
    if target_file is None:
        print(f"❌ 无法在 {DATA_ROOT} 找到 T0111 的 .mat 文件，请检查路径！")
        return
        
    print(f"🎯 选定原始物理文件: {target_file}")
    print(f"⏳ 正在加载并截取 0.1秒 (10,000,000 点) 的 I/Q 数据...")
    
    try:
        with h5py.File(target_file, 'r') as f:
            i_data = f['RF0_I'][0, :NUM_SAMPLES]
            q_data = f['RF0_Q'][0, :NUM_SAMPLES]
            signal_clean = i_data + 1j * q_data
    except Exception as e:
        print(f"❌ 读取 .mat 文件失败: {e}")
        return

    print("🌌 正在应用极度恶劣的目标域信道模型...")
    signal_noisy = apply_extreme_channel(signal_clean, FS)

    print(f"📈 正在计算 Source Domain STFT (N={NPERSEG}, {WINDOW})...")
    f_mhz, t_s, mag_db_clean = compute_stft_db(signal_clean, FS, NPERSEG, WINDOW)
    plot_and_save(f_mhz, t_s, mag_db_clean, 
                  os.path.join(OUTPUT_DIR, "T0111_Matched_Source.png"), 
                  "T0111 - Source Domain (Clean Channel)")

    print(f"📉 正在计算 Target Domain STFT (N={NPERSEG}, {WINDOW})...")
    f_mhz, t_s, mag_db_noisy = compute_stft_db(signal_noisy, FS, NPERSEG, WINDOW)
    plot_and_save(f_mhz, t_s, mag_db_noisy, 
                  os.path.join(OUTPUT_DIR, "T0111_Matched_Target.png"), 
                  "T0111 - Target Domain (Extreme AWGN & Fading)")

    print(f"\n🎉 完美对齐的 STFT 对比图已生成完毕！请查看 {OUTPUT_DIR} 目录。")

if __name__ == '__main__':
    main()