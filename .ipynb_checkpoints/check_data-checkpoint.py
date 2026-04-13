import os
from collections import Counter
from config import config

# 【修复】：把原来的解析逻辑直接搬到这里，自给自足！
def parse_filename(filepath):
    basename = os.path.basename(filepath).replace('.mat', '')
    parts = basename.split('_')
    t_code = parts[0]
    
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

def analyze_dataset():
    data_dir = config.DATA_ROOT
    print(f"🔍 正在扫描原始数据目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("❌ 错误：找不到该目录，请检查 config.DATA_ROOT 的路径设置。")
        return

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    print(f"📂 总计找到 {len(all_files)} 个 .mat 文件\n")
    
    counts = Counter()
    for f in all_files:
        label = parse_filename(f)
        counts[label] += 1
        
    print(f"{'类别 ID':<8} | {'文件总数':<8} | {'预计进入测试集 (15%)':<18} | {'状态诊断'}")
    print("-" * 75)
    
    for label in range(25):
        total = counts.get(label, 0)
        
        # 1:1 模拟切分逻辑
        n_train = int(total * 0.7)
        n_val = int(total * 0.15)
        
        if n_train == 0 and total >= 3: n_train = 1
        if n_val == 0 and total >= 2: n_val = 1
        
        n_test = total - n_train - n_val
        n_test = max(0, n_test) 
        
        status = "✅ 健康"
        if total == 0:
            status = "❌ 完全缺失 (请检查数据源)"
        elif total < 5:
            status = "⚠️ 长尾极少数类"
        
        if total > 0 and n_test == 0:
            status += " 🚨 测试集分配为 0 (触发报错元凶)!"
            
        print(f"Class {label:02d} | {total:<10} | {n_test:<20} | {status}")

if __name__ == '__main__':
    analyze_dataset()