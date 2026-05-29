import sys
import os
os.environ['TIMM_FUSED_ATTN'] = '0' if 'TIMM_FUSED_ATTN' not in os.environ else os.environ['TIMM_FUSED_ATTN']
import os.path as osp
from time import time as ttime
import argparse
import random
from collections import OrderedDict
import tqdm
from typing import Any, Literal
from copy import deepcopy
import warnings
import scipy.ndimage
from typing import Dict, List
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import open_clip
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.cuda.amp.grad_scaler import GradScaler
import torch.utils.hooks
from torch.utils.data import DataLoader, TensorDataset
import torch.linalg
import torchvision
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2



from utils.mod_adam import ModAdam
import utils.vit_builder
from utils.vit_builder import VisionTransformer
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset, Mixup, define_dataset
from utils.continual_manager import ClassIncrementalManager
from utils import misc
import clip
from clip.clip import CLIP
from einops import rearrange, reduce, repeat

torch.set_float32_matmul_precision("high")


class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[Literal['base_params', 'task_params_'], OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, tuple[int, int, int]]
    text_features_cache: Tensor = None

    def init_from_args(self, args):
        self.args = args
        _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
        self.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                               'eval': _dataset_class_manager[args.dataset](train=False)}
        self.cl_mngr = ClassIncrementalManager(self.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes)
        self.acc_mat_dict = OrderedDict(AccClassIncMat=np.zeros([_nt := self.cl_mngr.num_tasks, _nt]), AccClassIncList=np.zeros([_nt]))
        self.cache_dict = {}
        self.param_dict = {}
        self.label_map_g2l = {}

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map = misc.make_label_maps(taskid, task_classes)
        if not all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()]):
            print("The global_to_local label map has been fully loaded, which is not expected.")
        self.label_map_g2l.update(_g2l_map)
        return _g2l_map


def get_args():
    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'sdomainet','cub200', 'imagenet100'), help='use lowercase')
    parser.add_argument('-dr', '--data_root', type=str, default="")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 10, 20, 25, 50, 100))
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-m', '--model', type=str, default='vit_base_patch16_224.augreg_in21k', help='vit_base_patch16_224.augreg_in21k, vit_base_patch16_clip_quickgelu_224.openai')
    parser.add_argument('--head_dim_type', type=str, choices=('task_classes', 'pretrained', 'text_dim'), default='task_classes')
    parser.add_argument('--logit_type', type=str, choices=('head_out', 'sim_imgtext'), default='head_out')
    parser.add_argument('--logit_scale', type=float, default=4.605170249938965, help='0 | 4.605170249938965')
    parser.add_argument('--logit_scale_trainable', type=misc.str2bool, default=False)
    parser.add_argument('--prompt_len', type=int, default=0, help='0 means not using prompt')
    parser.add_argument('--prompt_init', type=str, choices=('uniform', 'zero'), default='uniform')
    parser.add_argument('--prompt_start_block', type=int, default=0)
    parser.add_argument('--prompt_end_block', type=int, default=3)
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)
    parser.add_argument('--use_null_space', action='store_true')
    parser.add_argument('--null_patterns', type=str, nargs='+', default=('lora_',))
    parser.add_argument('--null_thres_mode', type=str, choices=('adaptive', 'times'), default='adaptive')
    parser.add_argument('--null_thres_value1', type=float, default=0.)
    parser.add_argument('--null_thres_value2', type=float, default=0.)
    parser.add_argument('--null_eta1', type=float, default=0.98)
    parser.add_argument('--null_eta2', type=float)
    parser.add_argument('--null_interm_accum', type=str, choices=('sum', 'mean'), default='sum')
    parser.add_argument('--ln_loss_lam', type=float, default=1.)
    parser.add_argument('--refine_head', type=misc.str2bool)
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug', 'prototype', 'clip', 'siglip'), default='autoaug')
    parser.add_argument('--prob_cutmixup', type=float, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)
    parser.add_argument('-et', '--expand_times', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=28.)
    parser.add_argument('--temperature_trainable', type=misc.str2bool, default=False)
    parser.add_argument('--use_amp', type=misc.str2bool, default=True)
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--consecutive_training', type=misc.str2bool, default=True, help="")
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    parser.add_argument('--training_string', type=str, nargs='+')
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_scale', type=float)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+')
    parser.add_argument('--lr_scale_2', type=float, help='Second learning rate scale factor')
    parser.add_argument('--lr_scale_patterns_2', type=str, nargs='+', help='Patterns for third learning rate group')
    parser.add_argument('--optimizer', type=str, default='mod_adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5, 8])
    parser.add_argument('--decay_epochs', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--show_bar', action='store_true')
    parser.add_argument('--print_model', action='store_true')
    parser.add_argument('--lora_r', default=4, type=int, help='0 means not using lora')
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_blocks', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--use_margin', type=misc.str2bool, default=False)
    parser.add_argument('--use_random_margin_value', type=misc.str2bool, default=False)
    parser.add_argument('--margin_value', type=float, default=1.)
    parser.add_argument('--finetune_margin', type=misc.str2bool, default=True)
    parser.add_argument('--train_all_margins', type=misc.str2bool, default=False, help='train all margin params instead of freezing previous tasks')
    parser.add_argument('--refinehead_margin', type=misc.str2bool, default=False)
    parser.add_argument('--rh_margin_lr', type=float, default=0.0001)
    parser.add_argument('--margin_loss_lam', type=float, default=0.1)
    parser.add_argument('--l2_loss_lam', type=float, default=0., help='L2 regularization coefficient')
    parser.add_argument('--epochs_rh', type=int, default=50)
    parser.add_argument('--tsne_bac', action='store_true')
    parser.add_argument('--tsne_taskid', type=int, default=1)
    parser.add_argument('--tsne_samples_per_class', type=int, default=100)
    parser.add_argument('--tsne_num_classes', type=int, default=10)


    args = parser.parse_args()

    if args.null_eta2 is None:
        args.null_eta2 = args.null_eta1

    if args.logit_type == 'head_out':
        if args.refine_head is None:
            args.refine_head = False
        if args.training_string is None:
            args.training_string = ['head', 'lora', 'margin']
        if args.lr_scale_patterns is None:
            args.lr_scale_patterns = ['lora']
        if args.lr_scale is None:
            args.lr_scale = 0.2
        if args.lr_scale_patterns_2 is None:
            args.lr_scale_patterns_2 = ['margin']
        if args.lr_scale_2 is None:
            args.lr_scale_2 = 0.02
    else:
        if args.refine_head is None:
            args.refine_head = True
        if args.training_string is None:
            args.training_string = ['lora', 'head', 'logit_scale', 'margin']
        if args.lr_scale_patterns is None:
            args.lr_scale_patterns = []
        if args.lr_scale is None:
            args.lr_scale = 1.
        if args.lr_scale_patterns_2 is None:
            args.lr_scale_patterns_2 = []
        if args.lr_scale_2 is None:
            args.lr_scale_2 = 1.

    if args.optimizer not in ('mod_adam',):
        raise NotImplementedError(args.optimizer)

    # if not args.use_null_space:
    #     if args.ln_loss_lam != 0:
    #         print("args.ln_loss_lam is set to 0 for not using null space.")
    #     args.ln_loss_lam = 0

    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    torchvision.set_image_backend('accimage')


def set_model_mode(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, to_gpu: bool = True, training_string: tuple[str] = ('prompt',)) -> VisionTransformer:
    for n, p in model.named_parameters():
        if training and any([_s in n for _s in training_string]):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    model.eval()
    for n, m in model.named_modules():
        if training and any([n.endswith(_s) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]

    if to_gpu:
        model.cuda()

    if training:
        for n in GVM.cache_dict['not_pretrained_params']:
            if n not in params_requires_grad:
                raise ValueError(f"'{n}' does not require grad but it is not pretrained.")
    else:
        assert len(params_requires_grad) == 0, f"{params_requires_grad}"
        assert len(modules_training) == 0, f"{modules_training}"

    return model

def set_learning_rates(GVM: GlobalVarsManager, model: VisionTransformer, base_lr: float, lr_scale: float, lr_scale_2: float, lr_scale_patterns: list[str], lr_scale_patterns_2: list[str]) -> list[dict[str, Tensor | float]]:
    # 创建三个参数组：基础学习率、缩放学习率1、缩放学习率2
    param_lr_groups = [
        {'params': [], 'lr': base_lr},                    # 第一组：基础学习率
        {'params': [], 'lr': base_lr * lr_scale},         # 第二组：基础学习率 * 缩放因子
        {'params': [], 'lr': base_lr * lr_scale_2}
    ]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for n, p in model.named_parameters():
        if p.requires_grad:
            # 根据参数名称匹配不同的模式来分配到不同组
            if any(pattern in n for pattern in lr_scale_patterns):
                # 匹配第一组缩放模式的参数
                _group_idx = 1
            elif any(pattern in n for pattern in lr_scale_patterns_2):
                # 匹配第二组缩放模式的参数（需要在args中添加对应的参数）
                _group_idx = 2
            else:
                # 其余参数使用基础学习率
                _group_idx = 0

            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)

    # 移除空的参数组
    param_lr_groups = [group for group in param_lr_groups if len(group['params']) > 0]

    return param_lr_groups


def train_one_epoch(GVM: GlobalVarsManager, curr_epoch: int, prev_classes: int, dataloader: DataLoader, model: VisionTransformer, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer) -> str:
    args = GVM.args

    # temperature: Tensor = model.temperature if hasattr(model, 'temperature') else torch.tensor(args.temperature,device=model.device)
    use_amp: bool = args.use_amp
    # assert temperature > 0.
    if not args.use_null_space:
        assert args.ln_loss_lam == 0
    # else:
    #     assert args.ln_loss_lam == 1
    _use_cutmixup = args.prob_cutmixup > 0

    if _use_cutmixup:
        cutmixup_fn = Mixup(mixup_alpha=1., cutmix_alpha=1., prob=args.prob_cutmixup, switch_prob=0.5, mode='batch', num_classes=len(GVM.cl_mngr.current_task_classes))

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%")
    _btimer = ttime()

    if GVM.cl_mngr.current_taskid > 0 and args.use_margin:
        # weight_M1 = model.state_dict()['head.weight'][ :len(GVM.cl_mngr.sofar_task_classes)-len(GVM.cl_mngr.current_task_classes), : ]
        # # M2 = GVM.param_dict[f'task_params_{GVM.cl_mngr.current_taskid + 1}']['head.weight']
        # weight_M2 = model.state_dict()['head.weight'][ -len(GVM.cl_mngr.current_task_classes):, : ]
        if GVM.cl_mngr.current_taskid != 1:
            weight_M1 = GVM.cache_dict['head_weight'][ :len(GVM.cl_mngr.sofar_task_classes)-len(GVM.cl_mngr.current_task_classes), :]
        else:
            weight_M1 = model.state_dict()['head.weight'][ :len(GVM.cl_mngr.sofar_task_classes)-len(GVM.cl_mngr.current_task_classes), : ]


        # print(f"head.shape:{GVM.cache_dict['head_weight'].shape}")

        # print(f"M1.shape: {weight_M1.shape}, M2.shape: {weight_M2.shape}")

    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        if _use_cutmixup:
            mix_img, mix_lbl = cutmixup_fn(images, target)
        else:
            mix_img = images
            mix_lbl = target

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits: Tensor = model(mix_img)

        if i_batch == 1:
            if args.seperate_head:
                # print(f"logits.shape: {logits.shape}")
                # print(f"task_classed: {GVM.cl_mngr.current_task_classes}")
                assert logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
            else:
                assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        ce_loss = criterion(logits / model.temperature, mix_lbl)
        # 增加正则化的loss，保留先前知识
        LN_mean_loss = torch.zeros_like(ce_loss)
        LN_std_loss = torch.zeros_like(ce_loss)
        # L2正则化
        L2_loss = torch.zeros_like(ce_loss)
        
        if GVM.cl_mngr.current_taskid > 0:
            _dst_tt = GVM.cl_mngr.current_taskid - 1
            for _n0, _p0 in GVM.param_dict[f'task_params_{_dst_tt}'].items():
                if 'lora' in _n0:
                    _p0 = _p0.detach()
                    _pt = model.get_parameter(_n0)
                    _mpt, _mp0 = _pt.mean(-1), _p0.mean(-1)
                    LN_mean_loss += F.l1_loss(_mpt, _mp0)
                    _spt, _sp0 = _pt.std(-1, unbiased=False), _p0.std(-1, unbiased=False)
                    LN_std_loss += F.l1_loss(_spt, _sp0)

        # 添加L2正则化
        if args.l2_loss_lam > 0:
            for n, p in model.named_parameters():
                if p.requires_grad and 'lora' in n:
                    L2_loss += torch.norm(p, p=2)

        if args.use_margin:
            constraint_loss = torch.tensor(0.0, device=model.device)
            if args.logit_type == 'head_out':
                if prev_classes > 0:  # 只有存在旧类时才计算约束
                    weight_M2 = model.head.weight[-len(GVM.cl_mngr.current_task_classes):, :]
                    alpha_new = model.margin[-1]
                    weight_M1 = F.normalize(weight_M1.detach(), p=2, dim=1)
                    weight_M2 = F.normalize(weight_M2.detach(), p=2, dim=1)
                    lambda_val = 5e-5 * torch.eye(weight_M1.shape[0], device=weight_M1.device)
                    Q = torch.linalg.inv(weight_M1 @ weight_M1.T + lambda_val) @ weight_M1
                    weight_M2_scaled = weight_M2 * alpha_new.unsqueeze(1)
                    # print(f"margin.shape: {alpha_new.unsqueeze(1).shape}, weight_M2.shape: {weight_M2.shape}")
                    constraint = Q @ weight_M2_scaled.T * model.temperature
                    norm_inf = torch.norm(constraint, p=float('inf'))
                    constraint_loss = torch.relu(norm_inf - 1)
                    # constraint_loss = torch.mean(torch.abs(constraint))

            elif args.logit_type == 'sim_imgtext':
                # 获取文本特征矩阵
                text_features = GVM.text_features_cache
                if prev_classes > 0:  # 只有存在旧类时才计算约束
                    # 旧类的文本特征
                    M1 = text_features
                    M1 = F.normalize(M1.detach(), p=2, dim=1)
                    # 新类文本特征
                    M2 = model.get_buffer("text_features")
                    M2 = F.normalize(M2, p=2, dim=1)
                    alpha_new = model.margin[-1]

                    lambda_val = 5e-5 * torch.eye(prev_classes, device=M1.device)
                    # Q = torch.linalg.inv(M1 @ M1.T + lambda_val) @ M1

                    Q = torch.linalg.inv(M1 @ M1.T + lambda_val) @ M1

                    M2_scaled = M2 * (alpha_new.unsqueeze(1))
                    constraint = Q @ M2_scaled.T * model.logit_scale

                    norm_inf = torch.norm(constraint, p=float('inf'))
                    constraint_loss = torch.relu(norm_inf - 1)
                    # if max_abs > 1:
                    #     count = count + 1

        if args.use_margin:
            loss: Tensor = ce_loss + (LN_mean_loss + LN_std_loss) * args.ln_loss_lam + args.margin_loss_lam * constraint_loss + L2_loss * args.l2_loss_lam
            # loss: Tensor = ce_loss + (LN_mean_loss + LN_std_loss) * args.ln_loss_lam
        else:
            loss: Tensor = ce_loss + (LN_mean_loss + LN_std_loss) * args.ln_loss_lam + L2_loss * args.l2_loss_lam


        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        acc_top1 = misc.calc_accuracy(logits, target, topk=(1,))[0]
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), loss=loss.item(), acc_top1=acc_top1, batch_time=batch_time)
        _btimer = ttime()


    # print(f"prev_classes: {prev_classes}")
    if prev_classes > 0 and args.use_margin:
        # print(f"M1 shape: {M1.shape}, M2 shape: {M2.shape}")
        # print(f"alpha_new: {alpha_new}")
        # constraint = Q @ M2_scaled.T * model.logit_scale
        # print(f"constraint matrix (sample):\n{constraint[0, :5]}")
        # norm_inf = torch.norm(constraint, p=float('inf'))
        print(f"constraint_loss: {constraint_loss.item():.4f}")
        # print(count)

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def cache_state(GVM: GlobalVarsManager, taskid: int, model: VisionTransformer):
    if taskid == 0:
        base_params = OrderedDict()
    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone()
        else:
            if taskid == 0:
                base_params[n] = p.clone()

    if taskid == 0:
        GVM.param_dict['base_params'] = base_params
    GVM.param_dict[f'task_params_{taskid}'] = task_params


def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: VisionTransformer, **kwargs) -> VisionTransformer:
    args = GVM.args

    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: VisionTransformer = set_model_mode(GVM, model, training=True, training_string=GVM.cache_dict['training_string'])
    model = modify_head(GVM, model, training=True, task_classes=task_classes)

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, expand_times=args.expand_times)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, timeout=args.timeout if args.workers > 0 else 0,
                            drop_last=args.prob_cutmixup > 0, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    current_classes = GVM.cl_mngr.sofar_task_classes
    current_task_classes = GVM.cl_mngr.current_task_classes
    prev_classes = len(current_classes) - len(current_task_classes)  # 计算已有类别数

    if args.use_random_margin_value:
        # 0-1随机值
        margin_value = torch.rand(1).item()
    else:
        margin_value = args.margin_value


    if args.use_margin:
        if prev_classes > 0:
            new_margin = nn.Parameter(
                torch.full((len(current_task_classes),), margin_value, device=model.device),
                requires_grad=True
            )
            model.margin.append(new_margin)

            if args.train_all_margins:
                for _p in model.margin:
                    _p.requires_grad_(True)
            else:
                for _p in model.margin[:-1]:
                    _p.requires_grad_(False)
        else:
            new_margin = nn.Parameter(
                torch.full((len(current_task_classes),), margin_value, device=model.device),
                requires_grad=False
            )
            model.margin.append(new_margin)
            if args.logit_type == 'sim_imgtext':
                GVM.text_features_cache = model.get_buffer("text_features")
        

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_2, args.lr_scale_patterns, args.lr_scale_patterns_2)

    if taskid == 0:
        GVM.cache_dict['update_proj_dict'] = {}
    if args.use_null_space:
        if taskid == 0:
            GVM.cache_dict['null_param_id_dict'] = get_param_id_dict(model, args.null_patterns)
            GVM.cache_dict['interm_tensor_dict'] = {}
        else:
            assert list(GVM.cache_dict['update_proj_dict'].keys()) == list(GVM.cache_dict['null_param_id_dict'].keys())
    else:
        assert GVM.cache_dict['update_proj_dict'] == {}

    

    if args.optimizer == 'mod_adam':
        optimizer = ModAdam(param_groups, GVM.cache_dict['update_proj_dict'], arg_dict={}, lr=args.lr, weight_decay=args.weight_decay, foreach=True)
    else:
        optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs

    if taskid > 0 and args.use_margin:
        if args.train_all_margins:
            for p in model.margin:
                assert p.requires_grad
        else:
            for p in model.margin[:-1]:
                assert not p.requires_grad
            assert model.margin[-1].requires_grad
        initial_margin_values = model.margin[-1].detach().clone()

    if not args.finetune_margin and args.use_margin:
        for param in model.margin:
            param.requires_grad_(False)
    
    torch.cuda.empty_cache()
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(GVM, epoch, prev_classes, dataloader, model, criterion, optimizer)
            print(f"Task [{taskid+1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr:=str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)

    # # 在训练任务结束后进行权重归一化
    # with torch.no_grad():
    #     model.head.weight = nn.Parameter(F.normalize(model.head.weight, p=2, dim=1))


    cache_state(GVM, taskid, model)

    if args.use_null_space and taskid + 1 < GVM.cl_mngr.num_tasks:
        new_interm_tensor_dict = get_interm_tensor_dict(GVM, model, GVM.cache_dict['null_param_id_dict'])
        GVM.cache_dict['interm_tensor_dict'] = accumulate_interm_tensor_dict(GVM, GVM.cache_dict['interm_tensor_dict'], new_interm_tensor_dict)
        GVM.cache_dict['update_proj_dict'] = get_update_projection_dict(GVM, GVM.cache_dict['null_param_id_dict'], GVM.cache_dict['interm_tensor_dict'])

    if taskid > 0 and args.use_margin:
        final_margin_values = model.margin[-1].detach().clone()

        # 计算变化量
        change_in_margin = final_margin_values - initial_margin_values

        # 打印变化量
        print("Change in margin values:", change_in_margin)
        print("Final margin values:", final_margin_values)

    extract_class_features(GVM, model)
    if args.refine_head:
        refine_head(GVM, model)
    if args.use_margin and taskid > 0 and args.logit_type == 'sim_imgtext':
        if args.refine_head:
            new_text_features = model.get_buffer("text_features")[prev_classes:] * model.margin[-1].unsqueeze(1)
        else:
            new_text_features = model.get_buffer("text_features") * model.margin[-1].unsqueeze(1)
        GVM.text_features_cache = torch.cat([GVM.text_features_cache, new_text_features], 0)
        print(GVM.text_features_cache.shape)


    model.remove_text_features()

    print(f"Task [{taskid+1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: VisionTransformer) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, model, training=False)

    # # 只对新增的类别权重进行归一化
    # if args.use_margin and train_taskid > 0:
    #     # 计算新增类别的索引范围
    #     prev_classes = len(GVM.cl_mngr.sofar_task_classes) - len(GVM.cl_mngr.get_classes(train_taskid))
    #     curr_classes = len(GVM.cl_mngr.get_classes(train_taskid))
    #
    #     # 只归一化当前任务新增的权重
    #     model.head.weight.data[prev_classes:prev_classes + curr_classes] = F.normalize(
    #         model.head.weight.data[prev_classes:prev_classes + curr_classes], p=2, dim=1
    #     )


    scalar_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    torch.cuda.empty_cache()
    for images, target in tqdm.tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                logits: Tensor = model(images)

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        _preds = logits.argmax(dim=1)

        acc_class_inc, _, _ = misc.calc_acc_topnn_dynamically(_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_class_inc=acc_class_inc)

    assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()

    print(f"Task [{train_taskid+1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid+1:>{len(_tt:=str(train_taskid+1))}}/{_tt}]: eval_time={ttime()-_ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)

    return result_dict


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: VisionTransformer):
    model = modify_head(GVM, model, training=False)

    # print(f'head.shape:{model.head.weight.shape}')nvida
    GVM.cache_dict['head_weight'] = model.head.weight.detach().clone()


    torch.cuda.empty_cache()
    average_acc_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
        eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid)
        one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model)
        GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
        average_acc_meter.add_step_value(**one_result_dict)
    model.remove_text_features()

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    acc_info_dict = {
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),
        'class_inc_avg_acc': float(np.mean(GVM.acc_mat_dict['AccClassIncList'][:current_taskid+1])),
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),
    }
    _formatter = misc.ScalarFormatter(sep=' | ', class_inc_last_acc=">6.2%", class_inc_avg_acc=">6.2%",  class_inc_last_forg=">6.2%")

    print(f":: ** Results of task [{current_taskid+1}]: [ {_formatter(**acc_info_dict)} ] **")
    print(f":: ** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")


def find_not_pretrained_params(args ,model: VisionTransformer, pretrained: bool = True, pretrained_cfg: dict[str, str] = None, extra_pretrained_params: list[str] = []) -> list[str]:
    assert isinstance(extra_pretrained_params, (list, tuple))

    if isinstance(model, CLIP):
        model_path = osp.join(os.path.expanduser("~/.cache/clip"), osp.basename(clip.clip._MODELS['ViT-B/16']))
        

        assert osp.exists(model_path), model_path
        pre_state_dict: OrderedDict[str, Tensor] = torch.jit.load(model_path).state_dict()
    else:
        assert pretrained_cfg is not None

        if 'open_clip' in pretrained_cfg.get('hf_hub_filename', ''):
            _filename = timm.models._hub.HF_OPEN_CLIP_WEIGHTS_NAME
        else:
            _filename = timm.models._hub.HF_WEIGHTS_NAME
        pre_state_dict: OrderedDict[str, Tensor] = timm.models.load_state_dict_from_hf(pretrained_cfg['hf_hub_id'], _filename)

        if 'visual.class_embedding' in pre_state_dict.keys():
            pre_state_dict = timm.models.vision_transformer._convert_openai_clip(pre_state_dict, model)

    not_pretrained_params = []
    # if args.append_lora:
    #     for n, p in model.named_parameters():
    #         if 'lora_' in n:
    #             not_pretrained_params.append(n)
    # else:
    for n, p in model.named_parameters():
        if n not in pre_state_dict.keys() or not pretrained:
            not_pretrained_params.append(n)
        else:
            if p.shape != pre_state_dict[n].shape:
                not_pretrained_params.append(n)

    for n in deepcopy(not_pretrained_params):
        for _p in extra_pretrained_params:
            if _p in n:
                not_pretrained_params.remove(n)

    return not_pretrained_params


def get_param_id_dict(model: VisionTransformer, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any(re.search(pattern, n) for pattern in patterns):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    # # 打印参数id
    # for p_id, p_info in param_id_dict.items():
    #     print(f"{p_id}: {p_info}")
    # 成功获取parameter id
    return param_id_dict


def get_text_features(GVM: GlobalVarsManager, model: CLIP, task_classes: list[int]) -> Tensor:
    dataset_name: str = GVM.args.dataset
    class_text_list: list[str] = [clip.text_prompt_dict[dataset_name]["classes"][c] for c in task_classes]
    tmpl_text_list: list[str] = clip.text_prompt_dict[dataset_name]["templates"]

    if GVM.args.transform_type == 'siglip':
        text_model = GVM.cache_dict['siglip_text_model']
        tokenizer = GVM.cache_dict['siglip_tokenizer']
        text_model.cuda()
        with torch.device('cuda'):
            with torch.no_grad():
                text_tokens = tokenizer([_t.format(_c) for _c in class_text_list for _t in tmpl_text_list]).to(next(text_model.parameters()).device)
                text_features = text_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                text_features: Tensor = reduce(text_features, '(c p) d -> c d', p=len(tmpl_text_list), c=len(class_text_list), reduction='mean')
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                text_features = text_features.detach()
        text_model.cpu()
        torch.cuda.empty_cache()
        return text_features

    model.cuda()
    with torch.device('cuda'):
        with torch.no_grad():
            text_tokens = clip.tokenize([_t.format(_c) for _c in class_text_list for _t in tmpl_text_list]).to(next(model.parameters()).device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features: Tensor = reduce(text_features, '(c p) d -> c d', p=len(tmpl_text_list), c=len(class_text_list), reduction='mean')
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features = text_features.detach()
    model.cpu()
    torch.cuda.empty_cache()

    return text_features


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type

    match args.logit_type:
        case 'sim_imgtext':
            assert head_dim_type in ('pretrained', 'text_dim')
        case 'head_out':
            assert head_dim_type in ('task_classes')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case 'text_dim':
            if args.transform_type == 'siglip':
                head_dim_arg_dict['num_classes'] = 768
            else:
                head_dim_arg_dict['num_classes'] = 512
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes

    if args.logit_type == 'sim_imgtext':
        _text_model = GVM.cache_dict['siglip_text_model'] if args.transform_type == 'siglip' else GVM.cache_dict['clip_model']
        model.cache_text_features(get_text_features(GVM, _text_model, _target_classes))

    elif args.logit_type == 'head_out':
        if model.head.out_features != len(_target_classes):
            _mh = deepcopy(model.head)
            _mdevice = _mh.weight.device
            _mdtype = _mh.weight.dtype
            model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
            model.head.requires_grad_(_mh.weight.requires_grad)

            if training:
                assert model.head.weight.requires_grad
            else:
                assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
                _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.weight.data.shape == _hw.shape
                model.head.weight.data = _hw

                if _mh.bias is not None:
                    _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                    assert model.head.bias.data.shape == _hb.shape
                    model.head.bias.data = _hb
    else:
        raise ValueError(args.logit_type)

    return model


def get_interm_tensor_dict(GVM: GlobalVarsManager, model: VisionTransformer, null_param_id_dict: dict) -> dict[int, Tensor]:
    interm_tensor_dict: dict[int, dict[str, Tensor]] = {}
    # 初始化4x12的权重存储结构（假设最大block层数为12）
    

    def _forward_hook(module: nn.Module, args: tuple[Tensor], output: Tensor,block_idx: int ):
        _pre_tokens = 197

        # 主要需要修改的地方，以及给vit_builder.py中，加上lora的标识。
        if isinstance(module, nn.Linear):
            block_idx = block_idx
            _pid = id(module.weight)
            _interm_tensor: Tensor = args[0]
            assert _pid in null_param_id_dict
            _mname = module._lora_type
            # print(_mname +'_'+ f"{block_idx}")
            if hasattr(module, '_is_lora') and module._is_lora:
                # 获取module的名字

                # 需要满足 x · {Delta}_A_q = 0
                if _mname == 'q_a':
                    # print(_interm_tensor.shape)
                    # torch.Size([236400, 64])
                    # 重新构造_mname,加上block的索引
                    _mname = f'{_mname}_{block_idx}'
                    assert _interm_tensor.shape[-1] == 64  # head_dim = dim / num_heads = 64
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                    # print("Xq_a的shape")
                    # print(_interm_tensor.shape)

                # 需要满足 x · A_q · {Delta}_B_q = 0
                if _mname == 'q_b':
                    # print("Xq_b的size:")
                    # print(_interm_tensor.shape)
                    # torch.Size([236400, 4])
                    _mname = f'{_mname}_{block_idx}'
                    # # 获取A_q的权重
                    # q_a = lora_weights['q_a'][block_idx].to(device=_interm_tensor.device)

                    # # 这里对齐的应该是r的size
                    # assert _interm_tensor.shape[-1] == 4
                    # 这里的_interm_tensor 实际上就等于 x·A_q，因为hook前的x 已经经过A_q层了
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                    # print("Xq_b的shape")
                    # print(_interm_tensor.shape)

                # 需要满足 x · {Delta}_A_v = 0
                if _mname == 'v_a':
                    _mname = f'{_mname}_{block_idx}'
                    assert _interm_tensor.shape[-1] == 64  # head_dim = dim / num_heads = 64
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                    # print("Xv_a的shape")
                    # print(_interm_tensor.shape)

                # 需要满足 x · A_v · {Delta}_B_v = 0
                if _mname == 'v_b':
                    _mname = f'{_mname}_{block_idx}'
                    # assert _interm_tensor.shape[-1] == 4
                    # _interm_tensor = _interm_tensor @ v_a
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                    # print("Xv_b的shape")
                    # print(_interm_tensor.shape)
                    

                if _pid not in interm_tensor_dict:
                    interm_tensor_dict[_pid] = {}
                if _mname not in interm_tensor_dict[_pid]:
                    interm_tensor_dict[_pid][_mname] = torch.zeros_like(_interm_tensor)
                interm_tensor_dict[_pid][_mname] += _interm_tensor

        elif isinstance(module, utils.vit_builder.IntermReader):
            _pid = module.dst_param_id
            _mname = module.module_name
            _interm_tensor: Tensor = args[0]


            # 他这里，interm_reader_1对应的是原文的亲和力阶段，需要满足x·W_k·{Delta}_P ,这个阶段做的事，首先是获取x，这里就是arg[0]
            # 接着是获取w_k，在进行一些转化，得到x·W_k，得到新的_interm_tensor,再转化为2维，方便后续协方差的计算。
            if _mname == 'interm_reader_1':
                # other_args一开始就在vit_builder.py中，所以直接用module.other_args['w_qkv']获取即可]
                # 如果我想获得lora矩阵的参数，我应该使用model.get_parameter('block.attn.qkv.weight')来获取，注意这里需要精确的知道参数的路径
                w_qkv = module.other_args['w_qkv'].detach()

                # w_k是qkv矩阵的k矩阵，即qkv矩阵的第2个矩阵，这里需要将qkv矩阵reshape成(3, 768, 768)，
                # 然后取第2个矩阵，再reshape成(12, 64, 768)，因为block有12层，最后repeat成(b, 12, 64, 768)
                w_k: Tensor = rearrange(w_qkv, '(n do) di -> n do di', n=3, do=768, di=768).unbind(0)[1]
                w_k = rearrange(w_k, '(h d) D -> h d D', h=12, d=64, D=768)
                w_k = repeat(w_k, 'h d D -> b h d D', b=_interm_tensor.shape[0])

                _interm_tensor = _interm_tensor[:, :, :_pre_tokens]
                assert _interm_tensor.shape[2] == _pre_tokens, f"{_interm_tensor.shape}"
                # 计算qkv矩阵的k矩阵和qkv矩阵的q矩阵的乘积，然后计算协方差
                _interm_tensor = _interm_tensor @ w_k
                _interm_tensor = rearrange(_interm_tensor, 'b h n d -> (b h n) d')
                # 计算协方差
                _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
            # 同理这里对应的是原文中的聚合阶段。x·{Delta}_P = 0
            if _mname == 'interm_reader_2':
                _interm_tensor = _interm_tensor[:, :, :_pre_tokens, _pre_tokens:]
                assert _interm_tensor.shape[-1] == GVM.args.prompt_len
                _interm_tensor = rearrange(_interm_tensor, 'b h n m -> (b h n) m')
                _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]

            assert _mname in ('interm_reader_1', 'interm_reader_2')
            if _pid not in interm_tensor_dict:
                interm_tensor_dict[_pid] = {}
            if _mname not in interm_tensor_dict[_pid]:
                interm_tensor_dict[_pid][_mname] = torch.zeros_like(_interm_tensor)
            interm_tensor_dict[_pid][_mname] += _interm_tensor
        else:
            raise NotImplementedError()
        

    _handle_list: list[torch.utils.hooks.RemovableHandle] = []

    if 'lora_weights' not in GVM.cache_dict:
        GVM.cache_dict['lora_weights'] = {
            'q_a': OrderedDict(),
            'q_b': OrderedDict(),
            'v_a': OrderedDict(),
            'v_b': OrderedDict()
        }
    for n, m in model.named_modules():
        is_lora_module = hasattr(m, '_is_lora') and m._is_lora
        if is_lora_module:
             # 解析block索引
            match = re.search(r'blocks\.(\d+)', n)
            if not match:
                assert False, f"{n}"
            idx = int(match.group(1))
            
            # 获取LoRA类型
            lora_type = m._lora_type  # 从模块自身获取
            lora_weights = GVM.cache_dict['lora_weights']
            lora_weights[lora_type][idx] = m.weight.detach().cpu()

    for n, m in model.named_modules():
        # 原始条件判断
        is_interm_reader = 'interm_reader' in n and isinstance(m, utils.vit_builder.IntermReader)
        # interm_reader_1 interm_reader_2 interm_reader_1 interm_reader_2 interm_reader_1 interm_reader_2 interm_reader_1 interm_reader_2，进行处理的模块数量
        is_lora_module = hasattr(m, '_is_lora') and m._is_lora
        if is_lora_module or is_interm_reader:
            # print(n)
            # 解析block索引
            match = re.search(r'blocks\.(\d+)', n)
            if not match:
                assert False, f"{n}"
            block_idx = int(match.group(1))
            # 闭包工厂函数捕获当前block_idx
            def make_hook_wrapper(current_block_idx):
                def _hook_wrapper(module, input_args, output):
                    return _forward_hook(module, input_args, output, current_block_idx)
                return _hook_wrapper

            # 注册正确的hook
            _handle_list.append(
                m.register_forward_hook(make_hook_wrapper(block_idx))
            )
    
    model = set_model_mode(GVM, model, training=False)
    torch.cuda.empty_cache()

    args = GVM.args
    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, use_eval_transform=True, expand_times=1, )
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    for img, _ in dataloader:
        with torch.no_grad():
            img: Tensor
            model(img.cuda(non_blocking=True))
    torch.cuda.empty_cache()


    for _h in _handle_list:
        _h.remove()

    assert len(interm_tensor_dict) > 0  ,"The dictionary 'interm_tensor_dict' must contain at least one tensor."
    assert list(interm_tensor_dict.keys()) == list(null_param_id_dict.keys()), f"{interm_tensor_dict.keys()}; {null_param_id_dict}"
    #  用来储存各个任务阶段的样本数量，方便后面去求加权平均
    if (_k := 'interm_sample_list') not in GVM.cache_dict:
        GVM.cache_dict[_k] = []
    GVM.cache_dict[_k].append(len(dataloader.dataset))

    return interm_tensor_dict

# 这段代码的作用，用来累加各个任务阶段的interm_tensor,相当于储存各个阶段deep feature，用来后续去求null space
def accumulate_interm_tensor_dict(GVM: GlobalVarsManager, cached_interm_tensor_dict: dict[int, dict[str, Tensor]], new_interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    assert len(new_interm_tensor_dict) > 0
    args = GVM.args
    # 如果cached_interm_tensor_dict为空，说明为第一次任务，则直接返回new_interm_tensor_dict,
    if cached_interm_tensor_dict == {}:
        merged_interm_tensor_dict = new_interm_tensor_dict
    else:
        # 判断cached_interm_tensor_dict和new_interm_tensor_dict的keys是否相等，如果不相等，则抛出异常
        assert (_lc := list(cached_interm_tensor_dict.keys())) == (_ln := list(new_interm_tensor_dict.keys())), f"{_lc}, {_ln}"
        merged_interm_tensor_dict: dict[int, Tensor] = {}
        for _pid in cached_interm_tensor_dict.keys():
            merged_interm_tensor_dict[_pid] = {}
            for _mname in cached_interm_tensor_dict[_pid].keys():
                _cached_tensor = cached_interm_tensor_dict[_pid][_mname]
                _new_tensor = new_interm_tensor_dict[_pid][_mname]
                assert _cached_tensor.shape == _new_tensor.shape

                match args.null_interm_accum:
                    case 'sum':
                        merged_interm_tensor_dict[_pid][_mname] = _cached_tensor + _new_tensor
                    case 'mean':
                        # 求加权平均的tensor
                        _num_list: list[int] = GVM.cache_dict['interm_sample_list']
                        # 根据样本量去给于先前任务tensor和当前任务tensor不同的权重。
                        # sum(_num_list[:-1])历史所有任务的样本总数（排除当前任务），_num_list[-1]当前新增任务的样本数，sum(_num_list)累积总样本数 = 历史样本 + 当前样本
                        merged_interm_tensor_dict[_pid][_mname] = sum(_num_list[:-1]) / sum(_num_list) * _cached_tensor + _num_list[-1] / sum(_num_list) * _new_tensor
                    case _:
                        raise ValueError()
    return merged_interm_tensor_dict


def get_update_projection_dict(GVM: GlobalVarsManager, null_param_id_dict: dict, interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    args = GVM.args

    update_proj_dict = {}
    torch.cuda.empty_cache()
    # 自适应阈值计算函数
    def adaptive_threshold(svals: torch.Tensor, offset: float = 0):
        points: np.ndarray = svals.cpu().numpy()
        assert points.ndim == 1
        if len(points) >= 128:
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            _delta = 1
            diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            _drop_ratio = 0.03
            drop_num = int(len(points) * _drop_ratio / 2)
            assert len(points) - drop_num >= 10
            valid_o2 = diff_o2[drop_num:-drop_num]
            thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
        else:
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        i_thres = np.arange(len(points))[points >= thres_val].max()
        if 0 <= offset < 1:
            i_thres = min(i_thres + int(offset * (len(points) - i_thres)), len(points) - 1)
        else:
            i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_thres:] = 1
        zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
        return zero_idx
    
    # 主处理逻辑
    for _pid in interm_tensor_dict.keys():
        update_proj_dict[_pid] = {}
        for _mname in interm_tensor_dict[_pid].keys():# 遍历每个模块类型
            # svd分解
            _, S, U_trans = torch.linalg.svd(interm_tensor_dict[_pid][_mname], full_matrices=True)
            S: Tensor
            U_trans: Tensor
            # 根据不同的模块类型选择不同的阈值，需要添加参数
            if _mname not in {'interm_reader_1', 'interm_reader_2'}:
                thres_value = args.null_thres_value1
            else:
                thres_value = {'interm_reader_1': args.null_thres_value1, 'interm_reader_2': args.null_thres_value2}[_mname]
            # 选择阈值模式
            match args.null_thres_mode:
                case 'times':
                    zero_idx = S <= S[-1] * int(thres_value)
                case 'adaptive':
                    zero_idx = adaptive_threshold(S, offset=thres_value) # 使用自适应算法
                case _:
                    raise ValueError(args.null_thres_mode)
            zero_idx: torch.BoolTensor
            assert torch.count_nonzero(zero_idx) > 0, f"{zero_idx}, {type(zero_idx)}, {torch.count_nonzero(zero_idx)}"
            # 投影矩阵构建
            U0 = U_trans[zero_idx] # 选择超过阈值的奇异向量
            B = U0.T @ U0  # 计算投影矩阵
            B = B / torch.norm(B)  # 归一化处理
            
            # 混合投影矩阵，需要添加参数
            if _mname not in {'interm_reader_1', 'interm_reader_2'}:
                null_eta: float = args.null_eta1
            else:
                null_eta: float = {'interm_reader_1': args.null_eta1, 'interm_reader_2': args.null_eta2}[_mname]  # 获取混合系数
            update_proj_dict[_pid][_mname] = null_eta * B.detach() + (1 - null_eta) * torch.eye(B.shape[0], device=B.device, dtype=B.dtype)  # 线性混合单位矩阵

    return update_proj_dict


def extract_class_features(GVM: GlobalVarsManager, model: VisionTransformer) -> None:
    model = set_model_mode(GVM, model, training=False)
    torch.cuda.empty_cache()

    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    feats = torch.empty([len(dataset), 768], dtype=torch.float32)
    label = torch.empty([len(dataset)], dtype=torch.long)

    smp_idx = 0
    for img, lbl in dataloader:
        with torch.no_grad():
            img: Tensor
            lbl: Tensor
            _feat = model.encode_image(img.cuda(non_blocking=True), pre_logits=True).cpu()
            for _f, _l in zip(_feat, lbl):
                feats[smp_idx] = _f  # [batch_size, 768]
                label[smp_idx] = _l
                smp_idx += 1
    assert smp_idx == len(dataset)
    torch.cuda.empty_cache()

    if GVM.args.refine_head:
        _mean_list = []
        _cov_list = []
        _class_list = []
        for _l in label.unique():
            _cls_feats = feats[label == _l]
            _mean_list.append(torch.mean(_cls_feats, dim=0, keepdim=False))
            _cov_list.append(torch.cov(torch.tensor(_cls_feats, dtype=torch.float64).T) + torch.eye(_cls_feats.shape[-1]) * 1e-4)
            _class_list.append(_l)
        _mean_list = torch.stack(_mean_list)
        _cov_list = torch.stack(_cov_list)
        _class_list = torch.stack(_class_list)

        _key = 'class_features'
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] = {'mean': _mean_list, 'cov': _cov_list, 'class': _class_list}
        else:
            GVM.cache_dict[_key]['mean'] = torch.cat([GVM.cache_dict[_key]['mean'], _mean_list])
            GVM.cache_dict[_key]['cov'] = torch.cat([GVM.cache_dict[_key]['cov'], _cov_list])
            GVM.cache_dict[_key]['class'] = torch.cat([GVM.cache_dict[_key]['class'], _class_list])
            assert len(GVM.cache_dict[_key]['mean']) == len(GVM.cache_dict[_key]['cov']) == len(GVM.cache_dict[_key]['class']) == len(GVM.cl_mngr.sofar_task_classes)
    return None


def run_tsne_bac(GVM: GlobalVarsManager, model: VisionTransformer, taskid: int, task_classes: list[int], samples_per_class: int) -> None:
    args = GVM.args
    model = set_model_mode(GVM, model, training=False)
    if args.logit_type == 'sim_imgtext':
        _text_model = GVM.cache_dict['siglip_text_model'] if args.transform_type == 'siglip' else GVM.cache_dict['clip_model']
        model.cache_text_features(get_text_features(GVM, _text_model, task_classes))

    task_ids = sorted({GVM.label_map_g2l[c][0] for c in task_classes})
    if len(task_ids) > 1:
        print(f"t-SNE classes from tasks: {[t + 1 for t in task_ids]}")

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    num_classes = len(task_classes)
    target_counts = torch.zeros(num_classes, dtype=torch.long)
    feats_list: list[Tensor] = []
    labels_list: list[Tensor] = []
    max_samples = samples_per_class * num_classes
    label_ids = [GVM.label_map_g2l[c][2] for c in task_classes]
    label_to_plot = {label_id: idx for idx, label_id in enumerate(label_ids)}
    skipped_labels = 0

    for images, target in dataloader:
        if target_counts.min().item() >= samples_per_class:
            break
        images = images.cuda(non_blocking=True)
        target = target.long()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                logits = model(images)
        logits = logits.float().cpu()
        target = target.cpu()
        for _logit, _lbl in zip(logits, target):
            _lbl_id = int(_lbl.item())
            _idx = label_to_plot.get(_lbl_id, None)
            if _idx is None:
                if 0 <= _lbl_id < num_classes:
                    _idx = _lbl_id
                else:
                    skipped_labels += 1
                    continue
            if _idx < 0 or _idx >= num_classes:
                skipped_labels += 1
                continue
            if target_counts[_idx].item() < samples_per_class:
                feats_list.append(_logit)
                labels_list.append(torch.tensor(_idx, dtype=torch.long))
                target_counts[_idx] += 1
                if len(labels_list) >= max_samples:
                    break
        if len(labels_list) >= max_samples:
            break

    if len(labels_list) < 2:
        print("t-SNE skipped: not enough samples.")
        if args.logit_type == 'sim_imgtext':
            model.remove_text_features()
        return
    if skipped_labels > 0:
        print(f"t-SNE warning: skipped {skipped_labels} samples due to label mismatch.")

    feats = torch.stack(feats_list).numpy()
    labels = torch.stack(labels_list).numpy()
    n_samples = len(labels)
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=perplexity, random_state=args.seed)
    emb = tsne.fit_transform(feats)

    out_dir = osp.join('runs', 'tsne')
    os.makedirs(out_dir, exist_ok=True)
    bac_tag = 'bac_on' if args.use_margin else 'bac_off'
    if len(task_ids) > 1:
        if task_ids == list(range(task_ids[0], task_ids[-1] + 1)):
            task_tag = f"tasks{task_ids[0] + 1}to{task_ids[-1] + 1}"
            title_task = f"tasks {task_ids[0] + 1}-{task_ids[-1] + 1}"
        else:
            task_tag = "tasks" + "_".join(str(t + 1) for t in task_ids)
            title_task = "tasks " + ", ".join(str(t + 1) for t in task_ids)
    else:
        _only_task = task_ids[0]
        task_tag = f"task{_only_task + 1}"
        title_task = f"task {_only_task + 1}"
    out_path = osp.join(out_dir, f"tsne_{task_tag}_{bac_tag}.png")

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab20', s=12, alpha=0.85)
    plt.title(f"t-SNE logits {title_task} ({bac_tag})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, ticks=range(num_classes))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"t-SNE saved to {out_path}")

    if args.logit_type == 'sim_imgtext':
        model.remove_text_features()


def refine_head(GVM: GlobalVarsManager, model: VisionTransformer):
    feats_mean: Tensor = GVM.cache_dict['class_features']['mean']
    feats_cov: Tensor = GVM.cache_dict['class_features']['cov']
    feats_class: Tensor = GVM.cache_dict['class_features']['class']
    assert len(feats_class.unique()) == len(GVM.cl_mngr.sofar_task_classes)

    stat_dataset = TensorDataset(feats_mean, feats_cov, feats_class)

    model = modify_head(GVM, model, training=False)
    mhead = model.head

    mhead.train()
    mhead.cuda()
    mhead.requires_grad_()
    print("Head层参数量:", sum(p.numel() for p in mhead.parameters()))

    if args.use_margin:

        current_classes = len(GVM.cl_mngr.sofar_task_classes)
        current_task_classes = GVM.cl_mngr.current_task_classes
        prev_classes = current_classes - len(current_task_classes)  # 计算已有类别数
        if args.refinehead_margin:
            if prev_classes> 0:
                model.margin[-1].requires_grad_(True)

        optimizer_params = [
            {'params': mhead.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
            {'params': model.margin, 'lr': args.rh_margin_lr, 'weight_decay': 1e-4}
        ]
    else:
        optimizer_params = [
            {'params': mhead.parameters(), 'lr': 0.001}
        ]

    optimizer = create_optimizer_v2(optimizer_params, opt='sgd', lr=0.001, weight_decay=1e-4, momentum=0.9)
    scheduler, num_epochs = create_scheduler_v2(optimizer, 'multistep', num_epochs=GVM.args.epochs_rh, decay_milestones=[999,], decay_rate=0.1)
    criterion = nn.CrossEntropyLoss().cuda()
    from torch.distributions.multivariate_normal import MultivariateNormal

    torch.cuda.empty_cache()
    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", acc_top1="samp_avg:>6.2%")

    if args.use_margin:
        initial_margin_values = model.margin[-1].detach().clone()

    for epoch in range(1, num_epochs + 1):
        scheduler.step(epoch)

        smp_inp = []
        smp_tgt = []
        assert len(stat_dataset) == len(GVM.cl_mngr.sofar_task_classes)
        _ns = 256
        for _cmean, _ccov, _cclass in stat_dataset:
            m = MultivariateNormal(_cmean.float(), _ccov.float())
            _smp = m.sample(sample_shape=(_ns,))
            smp_inp.append(_smp)
            smp_tgt.append(torch.as_tensor([_cclass,] * _ns, dtype=torch.long))
        smp_inp = torch.cat(smp_inp)
        smp_tgt = torch.cat(smp_tgt)

        train_data = TensorDataset(smp_inp, smp_tgt)
        assert len(train_data) == len(stat_dataset) * _ns
        dataloader = DataLoader(train_data, batch_size=256, shuffle=True)

        for inp, tgt in dataloader:
            out: Tensor = mhead(inp.cuda(non_blocking=True))
            if model.logit_type == 'head_out':
                logits = out
            elif model.logit_type == 'sim_imgtext':
                logits = model.forward_logits(out)
            loss: Tensor = criterion(logits, tgt.cuda(non_blocking=True))

            if args.use_margin:
                constraint_loss = torch.tensor(0.0, device=model.device)
                # 获取文本特征矩阵
                text_features = GVM.text_features_cache
                if prev_classes > 0:  # 只有存在旧类时才计算约束
                    # 旧类的文本特征
                    M1 = text_features
                    M1 = F.normalize(M1.detach(), p=2, dim=1)
                    # 新类文本特征
                    M2 = model.get_buffer("text_features")[prev_classes:]
                    M2 = F.normalize(M2, p=2, dim=1)
                    alpha_new = model.margin[-1]
                    
                    lambda_val = 5e-5 * torch.eye(prev_classes, device=M1.device)
                    Q = torch.linalg.inv(M1 @ M1.T + lambda_val) @ M1
                    
                    M2_scaled = M2 * (alpha_new.unsqueeze(1))
                    constraint = Q @ M2_scaled.T * model.logit_scale
                    
                    # max_abs = torch.max(torch.abs(constraint))
                    norm_inf = torch.norm(constraint, p=float('inf'))
                    constraint_loss = torch.relu(norm_inf - 1)

            if args.use_margin:
                total_loss = loss + 0.1 * constraint_loss  # β=0.1根据实验调整
                # total_loss = loss
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc_top1, = misc.calc_accuracy(logits.cpu(), tgt.cpu(), topk=(1, ))
            scalar_meter.add_step_value(len(inp), loss=loss.item(), acc_top1=acc_top1)
        if (epoch % 10 == 0 or epoch == num_epochs):
            print(f":: epoch [{epoch}/{num_epochs}]: {scalar_meter.format_outout(scalar_meter.update_epoch_average_value())}")

    if args.use_margin:
        final_margin_values = model.margin[-1].detach().clone()


        change_in_margin = final_margin_values - initial_margin_values

        # print("Change in margin values:", change_in_margin)
        # print("Final margin values:", final_margin_values)

    torch.cuda.empty_cache()


def _param_dict_to_cpu(param_dict: dict) -> dict:
    cpu_dict = {}
    for _k, _v in param_dict.items():
        if isinstance(_v, OrderedDict):
            cpu_dict[_k] = OrderedDict((n, p.detach().cpu()) for n, p in _v.items())
        else:
            cpu_dict[_k] = _v
    return cpu_dict


def save_checkpoint(GVM: GlobalVarsManager, model: VisionTransformer) -> None:
    args = GVM.args
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ns_flag = "NS1" if args.use_null_space else "NS0"
    ln_flag = "LN1" if args.ln_loss_lam != 0 else "LN0"
    bac_flag = "BAC1" if args.use_margin else "BAC0"
    fname = f"{ts}_{args.dataset}_{ns_flag}_{ln_flag}_{bac_flag}_seed{args.seed}.pth"
    out_dir = "check_point"
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, fname)
    ckpt = {
        "model_state": model.state_dict(),
        "param_dict": _param_dict_to_cpu(GVM.param_dict),
        "args": vars(args),
        "seed": args.seed,
        "num_tasks": GVM.cl_mngr.num_tasks,
    }
    torch.save(ckpt, out_path)
    print(f"Checkpoint saved to {out_path}")


if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)

    GVM = GlobalVarsManager()
    GVM.init_from_args(args)

    GVM.cache_dict['exp_start_time'] = ttime()
    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#'*90} Task: [{taskid+1}/{GVM.cl_mngr.num_tasks}] {'#'*90}")
        print(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        if not args.consecutive_training or taskid == 0:
            _prompt_args_dict = misc.get_specific_args_dict(args, 'prompt_')
            _other_args_dict = misc.get_specific_args_dict(args, 'logit_')
            _other_args_dict['temperature'] = args.temperature
            _other_args_dict['temperature_trainable'] = args.temperature_trainable
            _head_dim_arg_dict = get_head_dim_arg_dict(GVM, args)

            if args.logit_type == 'sim_imgtext':
                if args.transform_type == 'siglip':
                    _siglip_model_name = 'ViT-B-16-SigLIP'
                    _siglip_model, _siglip_preprocess_train, _siglip_preprocess = open_clip.create_model_and_transforms(
                        _siglip_model_name, pretrained='webli', device='cpu'
                    )
                    # Keep only text encoder to avoid extra visual tower memory usage.
                    _siglip_model.visual = nn.Identity()
                    _siglip_tokenizer = open_clip.get_tokenizer(_siglip_model_name)
                    GVM.cache_dict['siglip_text_model'] = _siglip_model
                    GVM.cache_dict['siglip_preprocess_train'] = _siglip_preprocess_train
                    GVM.cache_dict['siglip_preprocess'] = _siglip_preprocess
                    GVM.cache_dict['siglip_tokenizer'] = _siglip_tokenizer
                else:
                    _clip_model, _clip_preprocess = clip.load('ViT-B/16', device='cpu')
                    GVM.cache_dict['clip_model'] = _clip_model
                    GVM.cache_dict['clip_preprocess'] = _clip_preprocess

            model: VisionTransformer = timm.create_model(args.model, pretrained=True, pretrained_strict=False,
                                                         **_head_dim_arg_dict,
                                                         prompt_args_dict=_prompt_args_dict,
                                                         other_args_dict=_other_args_dict,
                                                         lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                                         lora_blocks=args.lora_blocks,
                                                         use_margin=args.use_margin)
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)

        if args.consecutive_training and taskid > 0:
            pass

        _not_pretrained_params = find_not_pretrained_params(args , model, pretrained_cfg=model.pretrained_cfg)
        GVM.cache_dict['not_pretrained_params'] = _not_pretrained_params
        # 仅计算未预训练参数中的可训练参数量（需确保这些参数确实被设置为可训练）
        # not_pretrained_trainable = sum(p.numel() for n, p in model.named_parameters() if n in _not_pretrained_params and p.requires_grad)
        # print(f"Trainable parameters in '_not_pretrained_params': {not_pretrained_trainable:,}")
        # for name in _not_pretrained_params:
        #     print(f"- {name}")
        # print("Trainable params:", [n for n, p in model.named_parameters() if p.requires_grad])
        # print("Head层参数量:", sum(p.numel() for p in model.head.parameters()))
        GVM.update_label_maps(taskid, current_task_classes)
        GVM.cache_dict['training_string'] = args.training_string
        # Allow freezing some unmatched params even if not loaded from pretrained weights.
        _freeze_allow = {'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias'}
        for _p in list(GVM.cache_dict['not_pretrained_params']):
            if _p in _freeze_allow and not any(_s in _p for _s in GVM.cache_dict['training_string']):
                GVM.cache_dict['not_pretrained_params'].remove(_p)
                print(f"{_p} is not in training_string; keep it frozen and skip check.")
        if args.transform_type == 'siglip':
            _before = len(GVM.cache_dict['not_pretrained_params'])
            GVM.cache_dict['not_pretrained_params'] = [
                _p for _p in GVM.cache_dict['not_pretrained_params']
                if any(_s in _p for _s in GVM.cache_dict['training_string'])
            ]
            _skipped = _before - len(GVM.cache_dict['not_pretrained_params'])
            if _skipped > 0:
                print(f"siglip pretrained check: skip {_skipped} frozen params.")
        misc.check_param_training(GVM.cache_dict['not_pretrained_params'], GVM.cache_dict['training_string'])



        model = train_one_task(GVM, taskid, current_task_classes, model)

        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)

    save_checkpoint(GVM, model)

    if args.tsne_bac:
        candidate_classes = list(GVM.cl_mngr.all_classes)
        if args.tsne_num_classes > 0 and len(candidate_classes) > args.tsne_num_classes:
            rng = np.random.default_rng(args.seed)
            tsne_task_classes = rng.choice(candidate_classes, size=args.tsne_num_classes, replace=False).tolist()
        else:
            tsne_task_classes = list(candidate_classes)
        print(f"t-SNE sampled classes ({len(tsne_task_classes)}): {tsne_task_classes}")
        run_tsne_bac(GVM, model, GVM.cl_mngr.num_tasks - 1, tsne_task_classes, args.tsne_samples_per_class)


