import os
import sys
import cv2
from hog import PersonDetector

def main(subset_dir):
    subset_dir = os.path.abspath(subset_dir)
    subset_dir_name = os.path.basename(subset_dir.rstrip('/'))
    out_root = os.path.join(os.path.dirname(subset_dir), f"{subset_dir_name}_hog_res")
    visual_dir = os.path.join(out_root, "visual")
    bbox_dir = os.path.join(out_root, "bbox")
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    img_dir = os.path.join(subset_dir, "images")
    model_path = "hog/hog_detector_model.pkl"  # 请根据实际路径修改

    # 加载HOG检测器
    detector = PersonDetector(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3)
    )
    detector.load_model(model_path)

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(img_dir, fname)
        original_image = cv2.imread(img_path)
        print(f"处理图片: {os.path.splitext(fname)[0]}")
        if original_image is None:
            print(f"无法读取图片: {img_path}")
            continue

        orig_h, orig_w = original_image.shape[:2]
        test_image = cv2.resize(original_image, (400, 256))
        scale_x = orig_w / 400.0
        scale_y = orig_h / 256.0

        # 检测
        detections = detector.detect(test_image, downscale=1.25, step_size=(9, 9), threshold=0.5)
        detections = detector.non_max_suppression(detections, overlap_thresh=0.3)

        # 坐标映射回原图
        detections_scaled = []
        for x, y, score, w, h in detections:
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            orig_w_box = int(w * scale_x)
            orig_h_box = int(h * scale_y)
            detections_scaled.append((orig_x, orig_y, score, orig_w_box, orig_h_box))

        # 可视化
        result_image = original_image.copy()
        bbox_lines = []
        for x, y, score, w_box, h_box in detections_scaled:
            cv2.rectangle(result_image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(result_image, f"{score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            bbox_lines.append(f"{x:.2f} {y:.2f} {w_box:.2f} {h_box:.2f}")

        # 保存visual
        out_visual_path = os.path.join(visual_dir, fname)
        cv2.imwrite(out_visual_path, result_image)

        # 保存bbox到txt
        bbox_txt_path = os.path.join(bbox_dir, f"{os.path.splitext(fname)[0]}_bboxes.txt")
        with open(bbox_txt_path, "w") as f:
            for line in bbox_lines:
                f.write(line + "\n")

        print(f"Saved: {out_visual_path}, {bbox_txt_path}")

    print("HOG检测完成，结果已保存到", out_root)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python eval_person_hog.py <subset_dir>")
        sys.exit(1)
    main(sys.argv[1])