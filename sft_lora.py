"""
LoRA Fine-tuning script for DETR on Person Detection Task
使用LoRA (Low-Rank Adaptation) 对DETR模型进行行人检测任务的微调
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA层实现: 在原始权重旁边添加低秩分解的可训练参数
    W = W0 + BA, where B is (d, r) and A is (r, k)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA的A和B矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # x @ lora_A.T @ lora_B.T = x @ (lora_B @ lora_A).T
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """
    将原始Linear层与LoRA层组合
    """
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)


class MultiheadAttentionWithLoRA(nn.Module):
    """
    为MultiheadAttention添加LoRA支持
    主要针对Q, K, V的投影矩阵
    """
    def __init__(self, attn: nn.MultiheadAttention, rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.attn = attn
        embed_dim = attn.embed_dim
        
        # 为q, k, v投影添加LoRA
        self.lora_q = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout)
        self.lora_k = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout)
        self.lora_v = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout)
        self.lora_out = LoRALayer(embed_dim, embed_dim, rank, alpha, dropout)
        
        # 冻结原始attention的权重
        for param in self.attn.parameters():
            param.requires_grad = False
    
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # 应用LoRA到query, key, value
        query_lora = query + self.lora_q(query)
        key_lora = key + self.lora_k(key)
        value_lora = value + self.lora_v(value)
        
        # 使用原始attention
        attn_output, attn_weights = self.attn(
            query_lora, key_lora, value_lora,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask
        )
        
        # 在输出上应用LoRA
        attn_output = attn_output + self.lora_out(attn_output)
        
        return attn_output, attn_weights


# ============================================================================
# Apply LoRA to DETR Model
# ============================================================================

def apply_lora_to_model(model, lora_rank=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    """
    将LoRA应用到DETR模型的指定模块
    
    Args:
        model: DETR模型
        lora_rank: LoRA的秩
        lora_alpha: LoRA的缩放因子
        lora_dropout: LoRA的dropout率
        target_modules: 要应用LoRA的模块列表，如果为None则使用默认配置
    """
    if target_modules is None:
        # 默认对transformer的所有attention和FFN层应用LoRA
        target_modules = [
            "transformer.encoder",
            "transformer.decoder",
            "class_embed",  # 分类头
            "bbox_embed",   # bbox回归头
        ]
    
    lora_params = []
    
    # 递归遍历模型，找到所有需要应用LoRA的层
    def replace_linear_with_lora(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 检查是否在目标模块中
            should_apply = any(target in full_name for target in target_modules)
            
            if should_apply and isinstance(child, nn.Linear):
                # 替换Linear层
                lora_linear = LinearWithLoRA(child, lora_rank, lora_alpha, lora_dropout)
                setattr(module, name, lora_linear)
                lora_params.extend(list(lora_linear.lora.parameters()))
                print(f"Applied LoRA to Linear: {full_name}")
                
            elif should_apply and isinstance(child, nn.MultiheadAttention):
                # 替换MultiheadAttention
                lora_attn = MultiheadAttentionWithLoRA(child, lora_rank, lora_alpha, lora_dropout)
                setattr(module, name, lora_attn)
                lora_params.extend(list(lora_attn.lora_q.parameters()))
                lora_params.extend(list(lora_attn.lora_k.parameters()))
                lora_params.extend(list(lora_attn.lora_v.parameters()))
                lora_params.extend(list(lora_attn.lora_out.parameters()))
                print(f"Applied LoRA to MultiheadAttention: {full_name}")
            else:
                # 递归处理子模块
                replace_linear_with_lora(child, full_name)
    
    replace_linear_with_lora(model)
    
    return lora_params


def get_args_parser():
    parser = argparse.ArgumentParser('DETR LoRA fine-tuning for person detection', add_help=False)
    
    # LoRA specific arguments
    parser.add_argument('--lora_rank', default=8, type=int,
                        help='Rank of LoRA decomposition')
    parser.add_argument('--lora_alpha', default=16, type=int,
                        help='Scaling factor for LoRA')
    parser.add_argument('--lora_dropout', default=0.1, type=float,
                        help='Dropout rate for LoRA layers')
    parser.add_argument('--lora_target_modules', nargs='+', default=None,
                        help='Target modules to apply LoRA (default: transformer, class_embed, bbox_embed)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights (recommended for LoRA fine-tuning)')
    
    # Training arguments
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate (higher than full fine-tuning since we only train LoRA)')
    parser.add_argument('--lr_backbone', default=0, type=float,
                        help='Learning rate for backbone (0 if frozen)')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs (fewer than full fine-tuning)')
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    # Model parameters
    parser.add_argument('--pretrained_weights', type=str, 
                        default='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                        help="Path to pretrained DETR model")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    # Dataset parameters  
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True,
                        help='Path to COCO dataset (or person-only subset)')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--person_only', action='store_true',
                        help='Filter dataset to only include person class')
    
    parser.add_argument('--output_dir', default='./output_lora',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build model
    print("Building DETR model...")
    model, criterion, postprocessors = build_model(args)
    
    # Load pretrained weights
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        if args.pretrained_weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained_weights, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
        
        # 加载预训练权重（可能需要调整类别数）
        model_state = checkpoint['model']
        
        # 如果是针对person-only任务，需要调整分类头
        if args.person_only:
            # COCO中person的类别ID是0 (在模型内部是1，因为0是background)
            # 我们创建一个2类的分类器：背景 + 人
            print("Adapting model for person-only detection...")
            
            # 对于class_embed，我们只保留person相关的权重
            # COCO的num_classes是91，person在索引1的位置
            old_class_embed_weight = model_state['class_embed.weight']
            old_class_embed_bias = model_state['class_embed.bias']
            
            # 创建新的2类分类器权重 (背景 + person)
            new_class_embed_weight = torch.zeros(2, old_class_embed_weight.size(1))
            new_class_embed_bias = torch.zeros(2)
            
            # 复制背景和person的权重
            new_class_embed_weight[0] = old_class_embed_weight[0]  # 背景
            new_class_embed_weight[1] = old_class_embed_weight[1]  # person (COCO中的索引)
            new_class_embed_bias[0] = old_class_embed_bias[0]
            new_class_embed_bias[1] = old_class_embed_bias[1]
            
            model_state['class_embed.weight'] = new_class_embed_weight
            model_state['class_embed.bias'] = new_class_embed_bias
        
        # 加载权重（strict=False以处理可能的维度不匹配）
        msg = model.load_state_dict(model_state, strict=False)
        print(f"Loaded pretrained model with message: {msg}")
    
    model.to(device)
    
    # Apply LoRA to the model
    print("\n" + "="*80)
    print("Applying LoRA to model...")
    print("="*80)
    lora_params = apply_lora_to_model(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )
    print("="*80)
    
    # Freeze backbone if specified
    if args.freeze_backbone:
        print("Freezing backbone weights...")
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # Count parameters
    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora_params = sum(p.numel() for p in lora_params)
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {n_total_params:,}")
    print(f"  Trainable parameters: {n_trainable_params:,} ({100*n_trainable_params/n_total_params:.2f}%)")
    print(f"  LoRA parameters: {n_lora_params:,} ({100*n_lora_params/n_total_params:.2f}%)")
    print()
    
    # Setup optimizer - only train LoRA parameters and task-specific heads
    param_dicts = []
    
    # LoRA parameters with higher learning rate
    lora_param_names = set()
    for n, p in model_without_ddp.named_parameters():
        if 'lora' in n and p.requires_grad:
            lora_param_names.add(id(p))
    
    lora_trainable_params = [p for p in model_without_ddp.parameters() 
                              if id(p) in lora_param_names]
    
    if lora_trainable_params:
        param_dicts.append({
            "params": lora_trainable_params,
            "lr": args.lr,
        })
        print(f"Added {len(lora_trainable_params)} LoRA parameters to optimizer with lr={args.lr}")
    
    # Other trainable parameters (like class_embed, bbox_embed if not frozen)
    other_params = [p for n, p in model_without_ddp.named_parameters() 
                    if p.requires_grad and id(p) not in lora_param_names]
    
    if other_params:
        param_dicts.append({
            "params": other_params,
            "lr": args.lr * 0.1,  # 稍低的学习率
        })
        print(f"Added {len(other_params)} other trainable parameters with lr={args.lr * 0.1}")
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    # Build datasets
    print("\nBuilding datasets...")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers)
    
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    
    output_dir = Path(args.output_dir)
    
    # Resume from checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    # Evaluation only
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors,
            data_loader_val, base_ds, device, args.output_dir
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, 
                               output_dir / "eval.pth")
        return
    
    # Training loop
    print("\n" + "="*80)
    print("Starting LoRA fine-tuning for person detection...")
    print("="*80)
    start_time = time.time()
    
    best_ap = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        
        # Save checkpoint
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_lora.pth']
            
            # Extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_lora_{epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                # 只保存LoRA相关的参数以节省空间
                lora_state_dict = {}
                for name, param in model_without_ddp.named_parameters():
                    if 'lora' in name or 'class_embed' in name or 'bbox_embed' in name:
                        lora_state_dict[name] = param
                
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),  # 完整模型
                    'lora_state_dict': lora_state_dict,  # 仅LoRA参数
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        # Evaluation
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        
        # Track best model
        current_ap = test_stats['coco_eval_bbox'][0]  # AP
        if current_ap > best_ap:
            best_ap = current_ap
            if args.output_dir:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'lora_state_dict': lora_state_dict,
                    'epoch': epoch,
                    'args': args,
                    'ap': current_ap,
                }, output_dir / 'best_lora_model.pth')
                print(f"Saved best model with AP: {current_ap:.3f}")
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_trainable_params,
            'n_lora_parameters': n_lora_params,
            'best_ap': best_ap,
        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log_lora.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # For evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest_lora.pth']
                    if epoch % 10 == 0:
                        filenames.append(f'lora_{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                 output_dir / "eval" / name)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f'Best AP achieved: {best_ap:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR LoRA fine-tuning for person detection', 
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


"""
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    sft_lora.py \
    --coco_path datasets/data \
    --output_dir ./output_lora_person \
    --lora_rank 8 \
    --lora_alpha 16 \
    --batch_size 4 \
    --epochs 50 \
    --lr 5e-4 \
    --freeze_backbone \
    --person_only
"""