import os

class Config:
    # 数据路径
    DATA_ROOT = "/root/autodl-tmp/DroneRFa"
    IMAGE_DATA_ROOT = '/root/autodl-tmp/DroneRFa_Images'
    
    # 模型参数
    NUM_CLASSES = 25  # 修正：1类背景 + 24类无人机
    IMG_SIZE = 224
    
    # 信号处理参数
    SAMPLE_RATE = 100e6
    SEGMENT_LENGTH = 100000
    STFT_NPERSEG = 256
    STFT_NOVERLAP = 128
    
    # 训练参数
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 系统参数
    NUM_WORKERS = 20
    DEVICE = "cuda"
    
    # 路径
    CHECKPOINT_DIR = "./checkpoints"
    RESULT_DIR = "./results"
    LOG_DIR = "./logs"
    
    # 类别名称（严格对应论文表3，共25类）
    CLASS_NAMES = [
        "Background",           # T0000
        "Phantom 3",            # T0001
        "Phantom 4 Pro",        # T0010
        "MATRICE 200",          # T0011
        "MATRICE 100",          # T0100
        "Air 2S",               # T0101
        "Mini 3 Pro",           # T0110
        "Inspire 2",            # T0111
        "Mavic Pro",            # T1000
        "Mini 2",               # T1001
        "Mavic 3",              # T1010
        "MATRICE 300",          # T1011
        "Phantom 4 Pro RTK",    # T1100
        "MATRICE 30T",          # T1101
        "AVATA",                # T1110
        "DJI Self-built",       # T1111
        "MATRICE 600 Pro",      # T10000
        "VBar",                 # T10001
        "FrSky X20",            # T10010
        "Futaba T6IZ",          # T10011
        "Taranis Plus",         # T10100
        "RadioLink AT9S",       # T10101
        "Futaba T14SG",         # T10110
        "Yunzhuo T12",          # T10111
        "Yunzhuo T10"           # T11000
    ]

config = Config()