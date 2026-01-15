import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from hog.hog import Hog_descriptor  # 根据你的项目结构调整import

def main(subset_dir):
    subset_dir = os.path.abspath(subset_dir)
    subset_dir_name = os.path.basename(subset_dir.rstrip('/'))
    out_dir = os.path.join(os.path.dirname(subset_dir), f"{subset_dir_name}_hog_feature_visual")
    os.makedirs(out_dir, exist_ok=True)

    img_dir = os.path.join(subset_dir, "images")

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # HOG特征提取
        hog = Hog_descriptor(gray, cell_size=8, bin_size=9)
        _, hog_image = hog.extract()

        # 保存HOG特征图
        plt.figure(figsize=(6, 3))
        plt.axis('off')
        plt.imshow(hog_image, cmap=plt.cm.gray)
        out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_hog.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {out_path}")

    print("HOG特征可视化完成，结果已保存到", out_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python visual_hog.py <subset_dir>")
        sys.exit(1)
    main(sys.argv[1])