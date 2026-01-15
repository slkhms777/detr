import os
import time
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
import cv2
from models import build_model
from main import get_args_parser
from hog import PersonDetector

def eval_detr(img_dir, model, postprocessors, device, max_images=5):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    times = []
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_images]
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        orig_img = Image.open(img_path).convert("RGB")
        w, h = orig_img.size
        img = transform(orig_img).unsqueeze(0).to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model(img)
            orig_target_sizes = torch.tensor([[h, w]], device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)[0]
        end = time.time()
        times.append(end - start)
    return times

def eval_hog(img_dir, detector, max_images=5):
    times = []
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_images]
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        original_image = cv2.imread(img_path)
        if original_image is None:
            continue
        orig_h, orig_w = original_image.shape[:2]
        test_image = cv2.resize(original_image, (400, 256))
        start = time.time()
        detections = detector.detect(test_image, downscale=1.25, step_size=(9, 9), threshold=0.5)
        detections = detector.non_max_suppression(detections, overlap_thresh=0.3)
        end = time.time()
        times.append(end - start)
    return times

def main():
    parser = argparse.ArgumentParser(description="评测DETR和HOG推理速度")
    parser.add_argument('--subset_dir', type=str, required=True, help='指定subset目录路径')
    args = parser.parse_args()

    subset_dir = os.path.abspath(args.subset_dir)
    img_dir = os.path.join(subset_dir, "images")
    subset_name = os.path.basename(subset_dir)

    # DETR模型加载（只加载一次权重，分别to不同device）
    detr_parser = get_args_parser()
    detr_args = detr_parser.parse_args([])
    detr_args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
    checkpoint = torch.hub.load_state_dict_from_url(detr_args.resume, map_location='cpu', check_hash=True)

    # GPU推理
    if torch.cuda.is_available():
        device_gpu = 'cuda'
        model_gpu, _, postprocessors_gpu = build_model(detr_args)
        model_gpu.load_state_dict(checkpoint['model'])
        model_gpu.eval()
        model_gpu.to(device_gpu)
        detr_times_gpu = eval_detr(img_dir, model_gpu, postprocessors_gpu, device_gpu, max_images=5)
    else:
        detr_times_gpu = []

    # CPU推理
    device_cpu = 'cpu'
    model_cpu, _, postprocessors_cpu = build_model(detr_args)
    model_cpu.load_state_dict(checkpoint['model'])
    model_cpu.eval()
    model_cpu.to(device_cpu)
    detr_times_cpu = eval_detr(img_dir, model_cpu, postprocessors_cpu, device_cpu, max_images=5)

    # HOG模型加载
    hog_model_path = "hog/hog_detector_model.pkl"
    detector = PersonDetector(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3)
    )
    detector.load_model(hog_model_path)
    hog_times = eval_hog(img_dir, detector, max_images=5)

    # 统计指标
    def summarize(times):
        times = [t for t in times if t > 0]
        if not times:
            return {"avg": 0, "min": 0, "max": 0}
        return {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "num": len(times)
        }

    detr_stats_gpu = summarize(detr_times_gpu)
    detr_stats_cpu = summarize(detr_times_cpu)
    hog_stats = summarize(hog_times)

    # 保存结果
    out_path = os.path.join("datasets/person_data", "speed_eval.txt")
    with open(out_path, "w") as f:
        f.write(f"评测subset: {subset_name}\n")
        f.write(f"图片数量: {detr_stats_cpu['num']}\n")
        if torch.cuda.is_available():
            f.write("DETR推理时间(GPU, s):\n")
            f.write(f"  平均: {detr_stats_gpu['avg']:.4f}\n")
            f.write(f"  最快: {detr_stats_gpu['min']:.4f}\n")
            f.write(f"  最慢: {detr_stats_gpu['max']:.4f}\n")
        else:
            f.write("DETR推理时间(GPU, s):\n  未检测到GPU\n")
        f.write("DETR推理时间(CPU, s):\n")
        f.write(f"  平均: {detr_stats_cpu['avg']:.4f}\n")
        f.write(f"  最快: {detr_stats_cpu['min']:.4f}\n")
        f.write(f"  最慢: {detr_stats_cpu['max']:.4f}\n")
        f.write("HOG推理时间(s):\n")
        f.write(f"  平均: {hog_stats['avg']:.4f}\n")
        f.write(f"  最快: {hog_stats['min']:.4f}\n")
        f.write(f"  最慢: {hog_stats['max']:.4f}\n")
    print(f"推理速度评测已保存到 {out_path}")

if __name__ == "__main__":
    main()