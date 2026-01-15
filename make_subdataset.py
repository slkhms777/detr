import os
import json
import random
from shutil import copy2
from PIL import Image, ImageDraw

# 配置路径
ANN_PATH = 'datasets/person_data/person_instances_val2017.json'
IMG_DIR = 'datasets/person_data/images'
OUT_DIR = 'datasets/person_data'
NUM_SUBSETS = 3
IMAGES_PER_SUBSET = 50
TOTAL_IMAGES = NUM_SUBSETS * IMAGES_PER_SUBSET
SEED = 42  # 固定种子

random.seed(SEED)

# 读取标注
with open(ANN_PATH, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']

# 按图片id分组标注，只保留person类别
ann_by_img = {}
for ann in annotations:
    if ann['category_id'] == 1:  # 只要人
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

# 随机选150张不同图片
all_img_infos = images.copy()
random.shuffle(all_img_infos)
selected_imgs = all_img_infos[:TOTAL_IMAGES]

# 分成三组
for i in range(NUM_SUBSETS):
    subset_imgs = selected_imgs[i*IMAGES_PER_SUBSET:(i+1)*IMAGES_PER_SUBSET]
    subset_name = f"subset{i+1}"
    subset_dir = os.path.join(OUT_DIR, subset_name)
    img_out_dir = os.path.join(subset_dir, "images")
    bbox_out_dir = os.path.join(subset_dir, "bbox")
    visual_out_dir = os.path.join(subset_dir, "visual")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(bbox_out_dir, exist_ok=True)
    os.makedirs(visual_out_dir, exist_ok=True)

    for img in subset_imgs:
        file_name = img['file_name']
        src_img_path = os.path.join(IMG_DIR, file_name)
        dst_img_path = os.path.join(img_out_dir, file_name)
        if not os.path.exists(dst_img_path):
            copy2(src_img_path, dst_img_path)

        # 保存bbox
        bbox_txt_path = os.path.join(bbox_out_dir, f"{os.path.splitext(file_name)[0]}_bboxes.txt")
        anns = ann_by_img.get(img['id'], [])
        with open(bbox_txt_path, "w") as f_bbox:
            for ann in anns:
                x, y, w, h = ann['bbox']
                f_bbox.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")

        # 可视化并保存
        img_pil = Image.open(src_img_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)
        for ann in anns:
            x, y, w, h = ann['bbox']
            draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
        visual_img_path = os.path.join(visual_out_dir, file_name)
        img_pil.save(visual_img_path)

    print(f"{subset_name} 完成，包含 {len(subset_imgs)} 张图片，已保存到 {subset_dir}")