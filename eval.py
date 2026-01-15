"""
Standalone evaluation script for DETR (no training dataset required).
Usage example:
python eval.py --batch_size 2 --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path datasets/data
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
from main import get_args_parser  # reuse main's parser for compatibility


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # load checkpoint (support url)
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    # build validation dataset and dataloader
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    # run evaluation
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                          data_loader_val, base_ds, device, args.output_dir)

    print("Evaluation finished. Test stats:")
    print(test_stats)

    # save eval results if requested
    if args.output_dir and utils.is_main_process():
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            torch.save(coco_evaluator.coco_eval["bbox"].eval, out_dir / "eval.pth")
        with (out_dir / "eval_log.txt").open("a") as f:
            f.write(str(test_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # ensure output dir exists if provided
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)