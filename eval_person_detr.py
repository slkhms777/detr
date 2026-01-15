import os
import sys
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from models import build_model
from main import get_args_parser

def main(subset_dir):
    # 目录配置
    subset_dir = os.path.abspath(subset_dir)
    subset_dir_name = os.path.basename(subset_dir.rstrip('/'))
    out_root = os.path.join(os.path.dirname(subset_dir), f"{subset_dir_name}_detr_res")
    visual_dir = os.path.join(out_root, "visual")
    bbox_dir = os.path.join(out_root, "bbox")
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    img_dir = os.path.join(subset_dir, "images")

    # 加载模型
    parser = get_args_parser()
    args = parser.parse_args([])  # 空参数，手动设置
    args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'  # 可替换为本地权重
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, postprocessors = build_model(args)
    checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(args.device)

    # 图像预处理（不resize，保持原图尺寸）
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(img_dir, fname)
        orig_img = Image.open(img_path).convert("RGB")
        w, h = orig_img.size
        img = transform(orig_img).unsqueeze(0).to(args.device)

        with torch.no_grad():
            outputs = model(img)
            orig_target_sizes = torch.tensor([[h, w]], device=args.device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)[0]

        # 只保留person类别（COCO类别id为1），且置信度合理
        person_mask = (results['labels'] == 1) & (results['scores'] > 0.7)
        boxes = results['boxes'][person_mask]
        scores = results['scores'][person_mask]

        draw = ImageDraw.Draw(orig_img)
        bbox_lines = []
        for box, score in zip(boxes, scores):
            x0, y0, x1, y1 = box.tolist()
            draw.rectangle([x0, y0, x1, y1], outline='blue', width=3)
            draw.text((x0, y0), f"{score:.2f}", fill='blue')
            # 转为 x, y, w, h
            w_box = x1 - x0
            h_box = y1 - y0
            bbox_lines.append(f"{x0:.2f} {y0:.2f} {w_box:.2f} {h_box:.2f}")

        # 保存visual
        out_visual_path = os.path.join(visual_dir, fname)
        orig_img.save(out_visual_path)

        # 保存bbox到txt
        bbox_txt_path = os.path.join(bbox_dir, f"{os.path.splitext(fname)[0]}_bboxes.txt")
        with open(bbox_txt_path, "w") as f:
            for line in bbox_lines:
                f.write(line + "\n")

        print(f"Saved: {out_visual_path}, {bbox_txt_path}")

    print("推理完成，结果已保存到", out_root)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python eval_person.py <subset_dir>")
        sys.exit(1)
    main(sys.argv[1])