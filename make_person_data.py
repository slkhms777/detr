import os
import json
from shutil import copy2

# COCO类别中person的类别id
PERSON_CATEGORY_ID = 1

# 路径配置
COCO_ANN_PATH = 'datasets/data/annotations/instances_val2017.json'  # 你也可以换成test2017
COCO_IMG_DIR = 'datasets/data/val2017'  # 或 'test2017'
OUT_DIR = 'datasets/person_data'
OUT_IMG_DIR = os.path.join(OUT_DIR, 'images')
OUT_ANN_PATH = os.path.join(OUT_DIR, 'person_instances_val2017.json')

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# 读取COCO标注
with open(COCO_ANN_PATH, 'r') as f:
    coco = json.load(f)

# 找出所有包含person的图片id
person_img_ids = set()
for ann in coco['annotations']:
    if ann['category_id'] == PERSON_CATEGORY_ID:
        person_img_ids.add(ann['image_id'])

# 筛选图片信息
person_images = [img for img in coco['images'] if img['id'] in person_img_ids]

# 筛选相关标注
person_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in person_img_ids]

# 构建新的json
person_coco = {
    'info': coco.get('info', {}),
    'licenses': coco.get('licenses', []),
    'images': person_images,
    'annotations': person_annotations,
    'categories': [cat for cat in coco['categories'] if cat['id'] == PERSON_CATEGORY_ID]
}

# 保存新的标注文件
with open(OUT_ANN_PATH, 'w') as f:
    json.dump(person_coco, f)

# 拷贝图片到新目录
for img in person_images:
    src_path = os.path.join(COCO_IMG_DIR, img['file_name'])
    dst_path = os.path.join(OUT_IMG_DIR, img['file_name'])
    if not os.path.exists(dst_path):
        copy2(src_path, dst_path)

print(f"共提取{len(person_images)}张含有人的图片，已保存到{OUT_IMG_DIR}")
print(f"相关标注已保存到{OUT_ANN_PATH}")