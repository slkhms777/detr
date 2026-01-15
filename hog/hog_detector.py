"""
HOG行人检测器复现
"""

import numpy as np
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
import glob
from skimage.transform import pyramid_gaussian
from .hog import Hog_descriptor
try:
    from imutils.object_detection import non_max_suppression
except ImportError:
    non_max_suppression = None


def sliding_window(image, window_size, step_size):
    """
    滑动窗口生成器
    """
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


class PersonDetector:
    """行人检测器"""
    
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        """
        初始化检测器
        
        Args:
            orientations: 方向直方图的bin数量
            pixels_per_cell: cell大小（像素）
            cells_per_block: 每个block包含的cell数量
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.svm = None
        self.window_size = (64, 128)  # 检测窗口大小
        
    def extract_hog_features(self, image):
        """
        使用手动实现的HOG提取特征
        
        Args:
            image: 输入图像（灰度或BGR格式）
            
        Returns:
            HOG特征向量
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 调整大小到统一尺寸
        gray = cv2.resize(gray, self.window_size)
        
        # 使用手动实现的HOG提取特征
        hog_descriptor = Hog_descriptor(
            gray,
            cell_size=self.pixels_per_cell[0],
            bin_size=self.orientations
        )
        hog_vector, _ = hog_descriptor.extract()
        
        # 展平为一维向量
        features = hog_vector.flatten()
        
        return features
    
    def load_training_data(self, positive_dir, negative_dir):
        """
        加载训练数据
        
        Args:
            positive_dir: 正样本目录
            negative_dir: 负样本目录
            
        Returns:
            features: HOG特征列表
            labels: 标签列表（1为正样本，0为负样本）
        """
        features = []
        labels = []
        
        # 加载正样本
        positive_files = glob.glob(os.path.join(positive_dir, "*.png"))
        print(f"找到 {len(positive_files)} 个正样本")
        
        for file_path in tqdm(positive_files, desc="提取正样本特征"):
            image = cv2.imread(file_path, 0)  # 直接读取为灰度图
            if image is not None:
                fd = cv2.resize(image, self.window_size)
                # 使用手动实现的HOG提取特征
                hog_descriptor = Hog_descriptor(
                    fd,
                    cell_size=self.pixels_per_cell[0],
                    bin_size=self.orientations
                )
                hog_vector, _ = hog_descriptor.extract()
                fd = hog_vector.flatten()
                features.append(fd)
                labels.append(1)
        
        # 加载负样本
        negative_files = glob.glob(os.path.join(negative_dir, "*.jpg"))
        print(f"找到 {len(negative_files)} 个负样本")
        
        for file_path in tqdm(negative_files, desc="提取负样本特征"):
            image = cv2.imread(file_path, 0)  # 直接读取为灰度图
            if image is not None:
                # 直接resize到统一尺寸
                fd = cv2.resize(image, self.window_size)
                # 使用手动实现的HOG提取特征
                hog_descriptor = Hog_descriptor(
                    fd,
                    cell_size=self.pixels_per_cell[0],
                    bin_size=self.orientations
                )
                hog_vector, _ = hog_descriptor.extract()
                fd = hog_vector.flatten()
                features.append(fd)
                labels.append(0)
        
        print(f"总共加载 {len(features)} 个样本")
        return np.float32(features), np.array(labels)
    
    def train(self, positive_dir, negative_dir):
        """
        训练检测器
        
        Args:
            positive_dir: 正样本目录
            negative_dir: 负样本目录
        """
        # 加载训练数据
        features, labels = self.load_training_data(positive_dir, negative_dir)
        
        # 训练Linear SVM
        print("训练Linear SVM...")
        self.svm = LinearSVC(C=1.0, max_iter=1000)
        self.svm.fit(features, labels)
        
        print("训练完成！")
    
    def detect(self, image, downscale=1.25, step_size=(9, 9), threshold=0.5):
        """
        在图像中检测行人（完全按照PersonDetection项目的实现）
        
        Args:
            image: 输入图像
            downscale: 尺度缩放因子
            step_size: 滑动窗口步长
            threshold: 决策阈值
            
        Returns:
            detections: 检测结果列表，每个检测结果包含(x, y, score, w, h)
        """
        if self.svm is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        detections = []
        scale = 0
        
        # 使用图像金字塔进行多尺度检测
        for im_scaled in pyramid_gaussian(image, downscale=downscale):
            # 如果缩放后的图像小于窗口大小，停止检测
            if im_scaled.shape[0] < self.window_size[1] or im_scaled.shape[1] < self.window_size[0]:
                break
            
            # 滑动窗口检测
            for (x, y, window) in sliding_window(im_scaled, self.window_size, step_size):
                # 检查窗口大小是否正确
                if window.shape[0] != self.window_size[1] or window.shape[1] != self.window_size[0]:
                    continue
                
                # 转换为灰度图
                if len(window.shape) == 3:
                    # 先将 float64 转换为 uint8，然后转换为灰度图
                    window_uint8 = (window * 255).astype(np.uint8)
                    window_gray = cv2.cvtColor(window_uint8, cv2.COLOR_BGR2GRAY)
                else:
                    # 单通道图像也需要转换为 uint8
                    window_gray = (window * 255).astype(np.uint8)
                
                # 提取HOG特征
                hog_descriptor = Hog_descriptor(
                    window_gray,
                    cell_size=self.pixels_per_cell[0],
                    bin_size=self.orientations
                )
                hog_vector, _ = hog_descriptor.extract()
                fd = hog_vector.flatten()
                
                fd = fd.reshape(1, -1)
                
                # 预测
                pred = self.svm.predict(fd)
                
                if pred == 1:
                    score = self.svm.decision_function(fd)[0]
                    
                    # 如果分数大于阈值，记录检测结果
                    if score > threshold:
                        # 将坐标转换回原始图像尺寸
                        orig_x = int(x * (downscale ** scale))
                        orig_y = int(y * (downscale ** scale))
                        orig_w = int(self.window_size[0] * (downscale ** scale))
                        orig_h = int(self.window_size[1] * (downscale ** scale))
                        
                        detections.append((orig_x, orig_y, score, orig_w, orig_h))
            
            scale += 1
        
        return detections
    
    def non_max_suppression(self, detections, overlap_thresh=0.3):
        """
        非极大值抑制，完全按照PersonDetection项目的实现
        
        Args:
            detections: 检测结果列表，每个检测结果包含(x, y, score, w, h)
            overlap_thresh: 重叠阈值
            
        Returns:
            抑制后的检测结果
        """
        if len(detections) == 0:
            return []
        
        # 转换为边界框数组
        rects = np.array([[x, y, x + w, y + h] for (x, y, score, w, h) in detections])
        sc = [score[0] if isinstance(score, np.ndarray) else score for (x, y, score, w, h) in detections]
        sc = np.array(sc)
        
        # 如果有imutils，使用它的NMS
        if non_max_suppression is not None:
            pick = non_max_suppression(rects, probs=sc, overlapThresh=overlap_thresh)
            
            # pick可能是索引数组，也可能是边界框数组
            if len(pick) > 0 and isinstance(pick[0], (int, np.integer)):
                # pick是索引数组
                return [detections[i] for i in pick]
            else:
                # pick是边界框数组
                result = []
                for box in pick:
                    # 找到匹配的检测结果
                    for detection in detections:
                        x, y, score, w, h = detection
                        if abs(x - box[0]) < 2 and abs(y - box[1]) < 2:
                            result.append(detection)
                            break
                return result
        else:
            # 使用自定义的NMS
            # 按分数排序
            indices = np.argsort(sc)[::-1]
            
            keep = []
            
            while len(indices) > 0:
                # 保留分数最高的框
                current = indices[0]
                keep.append(current)
                
                # 计算与其他框的重叠
                if len(indices) == 1:
                    break
                
                x1 = np.maximum(rects[current, 0], rects[indices[1:], 0])
                y1 = np.maximum(rects[current, 1], rects[indices[1:], 1])
                x2 = np.minimum(rects[current, 2], rects[indices[1:], 2])
                y2 = np.minimum(rects[current, 3], rects[indices[1:], 3])
                
                # 计算重叠面积
                w = np.maximum(0, x2 - x1)
                h = np.maximum(0, y2 - y1)
                overlap = (w * h) / (
                    (rects[indices[1:], 2] - rects[indices[1:], 0]) *
                    (rects[indices[1:], 3] - rects[indices[1:], 1])
                )
                
                # 保留重叠小于阈值的框
                indices = indices[1:][overlap < overlap_thresh]
            
            # 返回抑制后的检测结果
            return [detections[i] for i in keep]
    
    def save_model(self, model_path):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        model_data = {
            'svm': self.svm,
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'window_size': self.window_size
        }
        
        joblib.dump(model_data, model_path)
        print(f"模型已保存到 {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        model_data = joblib.load(model_path)
        
        self.svm = model_data['svm']
        self.orientations = model_data['orientations']
        self.pixels_per_cell = model_data['pixels_per_cell']
        self.cells_per_block = model_data['cells_per_block']
        self.window_size = model_data['window_size']
        
        print(f"模型已从 {model_path} 加载")


def main():
    # 数据路径
    base_dir = "/Users/slkgaw/Homework/图像处理/实验5/HOG/DATAIMAGE"
    positive_dir = os.path.join(base_dir, "positive")
    negative_dir = os.path.join(base_dir, "negative")
    
    # 创建检测器（使用skimage的HOG参数）
    detector = PersonDetector(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3)
    )
    
    # 训练模型
    print("开始训练模型...")
    detector.train(positive_dir, negative_dir)
    
    # 保存模型
    model_path = os.path.join(base_dir, "hog_detector_model.pkl")
    detector.save_model(model_path)
    
    print("\n完成！")


if __name__ == "__main__":
    main()