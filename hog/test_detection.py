#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HOG检测器
"""

import cv2
import os
from hog_detector import PersonDetector


def main():
    """主函数"""
    # 数据路径
    base_dir = "/Users/slkgaw/Homework/图像处理/实验5/HOG/DATAIMAGE"
    positive_dir = os.path.join(base_dir, "positive")
    model_path = os.path.join(base_dir, "hog_detector_model.pkl")
    
    # 创建检测器
    detector = PersonDetector(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3)
    )
    
    # 加载已训练的模型
    print("加载模型...")
    detector.load_model(model_path)
    
    # 测试检测（使用test_person.jpg）
    print("\n测试检测功能...")
    test_image_path = "test_person.jpg"
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(base_dir, "..", "test.jpg")
    if not os.path.exists(test_image_path):
        test_image_path = "test.jpg"
    
    if os.path.exists(test_image_path):
        # 读取原始图像
        original_image = cv2.imread(test_image_path)
        print(f"原始测试图像尺寸: {original_image.shape}")
        
        # 保存原始尺寸
        orig_h, orig_w = original_image.shape[:2]
        
        # 按照PersonDetection项目的实现，resize图像到(400, 256)
        test_image = cv2.resize(original_image, (400, 256))
        print(f"调整后的测试图像尺寸: {test_image.shape}")
        
        # 计算缩放比例
        scale_x = orig_w / 400.0
        scale_y = orig_h / 256.0
        
        # 检测
        print("开始检测...")
        detections = detector.detect(test_image, downscale=1.25, step_size=(9, 9), threshold=0.5)
        print(f"检测到 {len(detections)} 个候选框")
        
        # 非极大值抑制
        print("应用非极大值抑制...")
        detections = detector.non_max_suppression(detections, overlap_thresh=0.3)
        print(f"抑制后剩余 {len(detections)} 个检测框")
        
        # 将检测结果转换回原始图像的坐标
        detections_scaled = []
        for x, y, score, w, h in detections:
            # 将坐标和尺寸转换回原始图像的尺度
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            orig_w = int(w * scale_x)
            orig_h = int(h * scale_y)
            detections_scaled.append((orig_x, orig_y, score, orig_w, orig_h))
        
        # 在原始图像上绘制检测结果
        result_image = original_image.copy()
        for x, y, score, w, h in detections_scaled:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, f"{score:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果（原始分辨率）
        result_path = os.path.join(base_dir, "detection_result.jpg")
        cv2.imwrite(result_path, result_image)
        print(f"检测结果已保存到 {result_path}")
        
        # 打印检测结果
        print("\n检测结果（原始图像坐标）:")
        for i, (x, y, score, w, h) in enumerate(detections_scaled):
            print(f"  检测框 {i+1}: 位置=({x}, {y}), 大小=({w}x{h}), 分数={score:.4f}")
    else:
        print(f"测试图像不存在: {test_image_path}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()