"""
晶圆缺陷数据加载模块
支持两种模式：
1. 从 Kaggle 下载真实 WM-811K 数据（需要 Kaggle 账号）
2. 生成模拟晶圆数据（无需下载，适合快速验证流程）
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import os
from PIL import Image, ImageDraw


# ==================== 模拟数据生成器（无需下载，先跑通流程） ====================

def generate_wafer_map(size=64, defect_type=None):
    """
    生成一张模拟晶圆图
    像素值：0=背景, 1=正常晶粒, 2=缺陷晶粒
    """
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 2 - 2
    
    # 画圆形晶圆（正常区域=1）
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = dist_from_center <= radius
    img[mask] = 1
    
    if defect_type is None or defect_type == "none":
        return img
    
    # 根据缺陷类型画缺陷（像素值=2）
    if defect_type == "Center":
        # 中心缺陷
        c_mask = dist_from_center <= radius * 0.3
        img[c_mask] = 2
        
    elif defect_type == "Donut":
        # 环形缺陷
        d_mask = (dist_from_center >= radius * 0.4) & (dist_from_center <= radius * 0.6)
        img[d_mask] = 2
        
    elif defect_type == "Edge-Loc":
        # 边缘局部缺陷
        angle = np.random.uniform(0, 2 * np.pi)
        ex = int(center + radius * 0.85 * np.cos(angle))
        ey = int(center + radius * 0.85 * np.sin(angle))
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if 0 <= ex+dx < size and 0 <= ey+dy < size and mask[ey+dy, ex+dx]:
                    if np.random.random() > 0.3:
                        img[ey+dy, ex+dx] = 2
                        
    elif defect_type == "Edge-Ring":
        # 边缘环缺陷
        er_mask = (dist_from_center >= radius * 0.75) & (dist_from_center <= radius * 0.9)
        img[er_mask] = 2
        
    elif defect_type == "Loc":
        # 局部缺陷
        lx = np.random.randint(center - 10, center + 10)
        ly = np.random.randint(center - 10, center + 10)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                if 0 <= lx+dx < size and 0 <= ly+dy < size and mask[ly+dy, lx+dx]:
                    if np.random.random() > 0.2:
                        img[ly+dy, lx+dx] = 2
                        
    elif defect_type == "Random":
        # 随机缺陷
        n_defects = np.random.randint(20, 60)
        for _ in range(n_defects):
            rx = np.random.randint(0, size)
            ry = np.random.randint(0, size)
            if mask[ry, rx]:
                img[ry, rx] = 2
                
    elif defect_type == "Scratch":
        # 划痕缺陷
        start_angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(15, 30)
        for i in range(length):
            sx = int(center + i * np.cos(start_angle))
            sy = int(center + i * np.sin(start_angle))
            if 0 <= sx < size and 0 <= sy < size and mask[sy, sx]:
                img[sy, sx] = 2
                # 加粗划痕
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= sx+dx < size and 0 <= sy+dy < size and mask[sy+dy, sx+dx]:
                            img[sy+dy, sx+dx] = 2
                            
    elif defect_type == "Near-full":
        # 大面积缺陷
        nf_mask = dist_from_center <= radius * 0.85
        # 随机保留一些正常区域
        for y in range(size):
            for x in range(size):
                if nf_mask[y, x] and img[y, x] == 1:
                    if np.random.random() > 0.15:
                        img[y, x] = 2
    
    return img


class SyntheticWaferDataset(Dataset):
    """模拟晶圆数据集（无需下载，立即可用）"""
    
    DEFECT_TYPES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", 
                    "Loc", "Random", "Scratch", "Near-full", "none"]
    
    def __init__(self, num_samples=5000, img_size=64, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        
        # 生成标签（不平衡分布，模拟真实情况）
        self.labels = []
        weights = [0.15, 0.05, 0.18, 0.20, 0.14, 0.05, 0.08, 0.02, 0.13]
        self.labels = np.random.choice(len(self.DEFECT_TYPES), size=num_samples, p=weights)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        defect_type = self.DEFECT_TYPES[label]
        
        # 生成图像
        wafer_map = generate_wafer_map(self.img_size, defect_type)
        
        # 转为 RGB 图像（0=黑背景, 1=绿正常, 2=红缺陷）
        rgb_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        rgb_img[wafer_map == 0] = [0, 0, 0]       # 背景 - 黑
        rgb_img[wafer_map == 1] = [0, 200, 0]     # 正常 - 绿
        rgb_img[wafer_map == 2] = [255, 0, 0]     # 缺陷 - 红
        
        img = Image.fromarray(rgb_img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)


# ==================== 真实数据加载器（从 Kaggle 下载） ====================

class WM811KDataset(Dataset):
    """真实的 WM-811K 数据集加载器"""
    
    DEFECT_TYPES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", 
                    "Loc", "Random", "Scratch", "Near-full", "none"]
    
    def __init__(self, pkl_path, transform=None, max_samples=None):
        self.transform = transform
        
        print(f"正在加载数据: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        # 解析数据
        self.wafer_maps = []
        self.labels = []
        
        for item in data[:max_samples] if max_samples else data:
            if 'waferMap' in item and 'failureType' in item:
                wafer_map = item['waferMap']
                label = item['failureType']
                
                # 确保是 numpy 数组
                if not isinstance(wafer_map, np.ndarray):
                    wafer_map = np.array(wafer_map)
                    
                self.wafer_maps.append(wafer_map)
                self.labels.append(label)
                
        print(f"加载完成: {len(self.wafer_maps)} 张图像")
        
    def __len__(self):
        return len(self.wafer_maps)
    
    def __getitem__(self, idx):
        wafer_map = self.wafer_maps[idx]
        label = self.labels[idx]
        
        # 转为 RGB
        h, w = wafer_map.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_img[wafer_map == 0] = [0, 0, 0]
        rgb_img[wafer_map == 1] = [0, 200, 0]
        rgb_img[wafer_map == 2] = [255, 0, 0]
        
        img = Image.fromarray(rgb_img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)


def get_data_loaders(batch_size=32, use_synthetic=True, data_path=None):
    """获取训练和测试数据加载器"""
    
    # 数据增强和预处理
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if use_synthetic:
        # 使用模拟数据（无需下载，立即可训练）
        print("使用模拟晶圆数据...")
        train_dataset = SyntheticWaferDataset(num_samples=8000, transform=transform_train)
        test_dataset = SyntheticWaferDataset(num_samples=2000, transform=transform_test)
    else:
        # 使用真实数据（需要提前下载 .pkl 文件）
        if data_path is None or not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}\n"
                                  f"请从 Kaggle 下载 WM-811K 数据集: "
                                  f"https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        # 这里简化处理，实际使用需要更复杂的数据解析
        dataset = WM811KDataset(data_path, transform=transform_train)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)
    
    return train_loader, test_loader, SyntheticWaferDataset.DEFECT_TYPES