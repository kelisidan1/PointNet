import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ShapenetBinaryDataset(Dataset):
    def __init__(self, data_dir, class_labels, num_points=1024):
        self.data_dir = data_dir
        self.class_labels = class_labels
        self.num_points = num_points
        self.samples = []  # 存储点云文件路径和对应标签的列表

        # 遍历数据目录并加载数据
        for label, class_name in enumerate(class_labels):
            class_dir = os.path.join(data_dir, class_name)
            point_cloud_files = os.listdir(class_dir)
            for file_name in point_cloud_files:
                file_path = os.path.join(class_dir, file_name)
                self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]

        # 从文件加载点云数据（这里需要根据数据的格式进行加载）
        # 你可能需要使用第三方库（如NumPy）来加载点云数据
        # 下面是一个示例，假设点云数据是从文本文件加载的
        # 你需要根据实际情况自行调整
        with open(file_path, 'r') as file:
            point_cloud_data = np.loadtxt(file, max_rows=self.num_points)

        # 转换数据为PyTorch张量
        point_cloud_tensor = torch.tensor(point_cloud_data, dtype=torch.float32)

        return point_cloud_tensor, label

if __name__ == '__main__':
    dataset = ShapenetBinaryDataset(data_dir='data', class_labels=['airplane', 'car'])
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(len(dataset))
    # 从数据集中获取一个样本
    sample = test_data[1]
    print(sample)

    # 打印样本的点云数据和标签
    point_cloud, label = sample
    print("Point Cloud Shape:", point_cloud.shape)
    print("Label:", label)

    # 选择一个样本进行可视化
    sample = dataset[1]
    point_cloud, label = sample

    # 绘制点云
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Point Cloud Visualization')
    plt.savefig('b.png',dpi=300)
    plt.show()
