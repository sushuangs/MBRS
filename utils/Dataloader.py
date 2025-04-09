import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset


import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class MBRSDataset(Dataset):
    def __init__(self, root, H=256, W=256, extensions=('jpg', 'jpeg', 'png', 'bmp')):
        """
        参数:
            root (str): 数据集根目录（需包含按类别划分的子文件夹）
            H, W (int): 最终输出的图像尺寸
            extensions (tuple): 支持的图像文件扩展名
        """
        super().__init__()
        self.H = H
        self.W = W
        self.root = os.path.expanduser(root)
        self.extensions = {'.' + ext.lstrip('.').lower() for ext in extensions}
        
        # 初始化类别映射和样本列表
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

        # 数据增强流程（包含防止尺寸不足的Resize）
        self.transform = transforms.Compose([
            # transforms.Resize((int(H * 1.1), int(W * 1.1))),  # 保证后续裁剪安全
            transforms.RandomCrop((H, W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def _find_classes(self):
        """获取类别名称到索引的映射"""
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        return classes, {cls_name: i for i, cls_name in enumerate(classes)}
    
    def _is_valid_image(self, path):
        """检查是否为有效图像（尺寸、比例、文件格式）"""
        try:
            with Image.open(path) as img:
                # 检查文件格式
                if os.path.splitext(path)[1].lower() not in self.extensions:
                    return False
                
                # 检查图像尺寸是否满足最小要求
                min_h, min_w = self.H//2, self.W//2
                if img.height < min_h or img.width < min_w:
                    return False
                
                # 检查宽高比例是否合理
                ratio = img.width / img.height
                if ratio < 0.5 or ratio > 2.0:
                    return False
                
                return True
        except (IOError, OSError):
            return False
    
    def _make_dataset(self):
        """构建有效样本列表（预处理阶段过滤）"""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                if os.path.isfile(path) and self._is_valid_image(path):
                    samples.append((path, class_idx))
        
        return samples
    
    def __getitem__(self, index):
        """直接加载预验证过的样本"""
        path, target = self.samples[index]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception as e:
            # 如果出现意外错误，返回随机样本避免中断训练
            return self[torch.randint(0, len(self), (1,)).item()]
    
    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self):
        """辅助方法：获取类别分布统计"""
        from collections import Counter
        return Counter([label for _, label in self.samples])