import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 🚀 1. 论文级 matplotlib 渲染配置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',       
    'axes.labelsize': 16,         
    'axes.titlesize': 18,         
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'savefig.dpi': 300,           
    'savefig.bbox': 'tight'       
})

# ==========================================
# 🚀 2. 核心路径与精选类别配置
# ==========================================
SOURCE_DIR = "/root/autodl-tmp/DroneRFa_Images"       
TARGET_DIR = "/root/autodl-tmp/DroneRFa_Target_Images" 
OUTPUT_BASE_DIR = "./results/Paper_Figures"           

# 扩充为 20 个极具代表性的无人机与遥控器类别
# 包含密集 OFDM 图传与剧烈跳频 (FHSS) 信号
SELECTED_CLASSES = {
    "1": "Phantom_3",
    "2": "Phantom_4_Pro",
    "3": "MATRICE_200",
    "5": "Air_2S",
    "6": "Mini_3_Pro",
    "7": "Inspire_2",
    "8": "Mavic_Pro",
    "9": "Mini_2",
    "10": "Mavic_3",
    "13": "MATRICE_30T",
    "14": "AVATA",              # FPV 穿越机，信号极具动态性
    "16": "MATRICE_600_Pro",
    "17": "VBar",
    "18": "FrSky_X20",          # 纯遥控器，跳频明显
    "19": "Futaba_T6IZ",
    "20": "Taranis_Plus",
    "21": "RadioLink_AT9S",
    "22": "Futaba_T14SG",
    "23": "Yunzhuo_T12",
    "24": "Yunzhuo_T10"
}

def render_and_save_stft(gray_img_path, save_path, title_text):
    """
    读取黑白图，渲染为绚丽的伪彩色热力图，并保存为高分辨率论文插图
    """
    gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        print(f"❌ 警告: 找不到图片 {gray_img_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # cmap='jet' 将底噪压成深蓝，高能量（跳频/图传）映射为亮红/黄
    im = ax.imshow(gray_img, cmap='jet', aspect='auto', origin='lower')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Magnitude Intensity', rotation=270, labelpad=20)

    ax.set_title(title_text, fontweight='bold', pad=15)
    ax.set_xlabel('Time (Samples)')
    ax.set_ylabel('Frequency Bins')

    plt.savefig(save_path)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"🎨 开始生成 20 类论文级高清 STFT 对比图...\n")

    for class_id, class_name in SELECTED_CLASSES.items():
        class_out_dir = os.path.join(OUTPUT_BASE_DIR, class_name)
        os.makedirs(class_out_dir, exist_ok=True)
        
        source_class_dir = os.path.join(SOURCE_DIR, class_id)
        target_class_dir = os.path.join(TARGET_DIR, class_id)

        if not os.path.exists(source_class_dir) or not os.path.exists(target_class_dir):
            print(f"⚠️ 跳过 {class_name}: 数据集目录不完整。")
            continue

        source_files = [f for f in os.listdir(source_class_dir) if f.endswith('.png')]
        target_files = [f for f in os.listdir(target_class_dir) if f.endswith('.png')]

        if not source_files or not target_files:
            print(f"⚠️ 跳过 {class_name}: 目录下没有图片。")
            continue

        # 随机抽取图片，确保每次运行可能看到不同的切片
        chosen_source_file = random.choice(source_files)
        chosen_target_file = random.choice(target_files)

        source_img_path = os.path.join(source_class_dir, chosen_source_file)
        target_img_path = os.path.join(target_class_dir, chosen_target_file)

        out_source = os.path.join(class_out_dir, f"{class_name}_Source_Clean.png")
        title_source = f"{class_name.replace('_', ' ')} (Source Domain)"
        render_and_save_stft(source_img_path, out_source, title_source)

        out_target = os.path.join(class_out_dir, f"{class_name}_Target_Noisy.png")
        title_target = f"{class_name.replace('_', ' ')} (Target Domain: AWGN + Fading)"
        render_and_save_stft(target_img_path, out_target, title_target)

        print(f"✅ 已生成: {class_name}")

    print(f"\n🎉 20 个类别的对比图全部生成完毕！请前往 {OUTPUT_BASE_DIR} 查看打包成果。")

if __name__ == '__main__':
    main()