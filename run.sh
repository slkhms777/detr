# uv run eval.py --batch_size 2 --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path datasets/data
uv run eval_person_hog.py datasets/person_data/subset1
uv run eval_person_hog.py datasets/person_data/subset2
uv run eval_person_hog.py datasets/person_data/subset3