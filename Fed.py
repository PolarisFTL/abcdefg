import datetime
import os
from functools import partial
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SSAN
from utils.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_anchors, get_classes,
                         seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch
from config import *


def weighted_average_weights(state_dicts, data_sizes):
    avg_dict = OrderedDict()
    total = sum(data_sizes)
    weights = [size / total for size in data_sizes]
    for k in state_dicts[0].keys():
        avg_dict[k] = sum([w * sd[k] for w, sd in zip(weights, state_dicts)])
    return avg_dict


if __name__ == "__main__":
    seed_everything(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    ngpus_per_node = torch.cuda.device_count()
    client_loss_histories = {}

    # Distributed
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    class_names, num_classes = get_classes(classes_path)
    anchors, _ = get_anchors(anchors_path)
    global_model = SSAN(anchors_mask, num_classes, pretrained=pretrained)
    if not pretrained:
        weights_init(global_model)
    if model_path:
        pretrained_dict = torch.load(model_path, map_location=device)
        model_dict = global_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        global_model.load_state_dict(model_dict)
    global_model = global_model.to(device)

    # Loss Logger
    if local_rank == 0:
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        fed_log_dir = os.path.join(save_dir, f"fed_loss_{time_str}")
        os.makedirs(fed_log_dir, exist_ok=True)
        fed_loss_history = LossHistory(fed_log_dir, global_model, input_shape=input_shape)
    else:
        fed_loss_history = None

    # Federated Training Config
    client_paths = ['client1.txt', 'client2.txt', 'client3.txt']
    global_rounds = 100
    local_epochs = 3
    fed_momentum = 0.9
    momentum_state = None

    for rnd in range(global_rounds):
        print(f"\n===== Global Round {rnd+1}/{global_rounds} =====")
        local_states = []
        data_sizes = []

        for client_id, ann_path in enumerate(client_paths, 1):
            print(f"-> Client {client_id} training on {ann_path}")

            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()

            client_save_dir = os.path.join(save_dir, f"client{client_id}")
            os.makedirs(client_save_dir, exist_ok=True)

            yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask)
            ema = ModelEMA(local_model)

            if local_rank == 0:
                if client_id not in client_loss_histories:
                    local_log_dir = os.path.join(client_save_dir, "loss")
                    os.makedirs(local_log_dir, exist_ok=True)
                    client_loss_histories[client_id] = LossHistory(local_log_dir, local_model, input_shape=input_shape)
                loss_history = client_loss_histories[client_id]
            else:
                loss_history = None

            with open(ann_path, 'r', encoding='utf-8') as f:
                train_lines = f.readlines()
            with open(val_annotation_path1, 'r', encoding='utf-8') as f:
                val_lines1 = f.readlines()
            with open(val_annotation_path2, 'r', encoding='utf-8') as f:
                val_lines2 = f.readlines()

            batch_size = Unfreeze_batch_size
            nbs = 64
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            pg0, pg1, pg2 = [], [], []
            for k, v in local_model.named_modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter): pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or 'bn' in k: pg0.append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter): pg1.append(v.weight)

            optimizer = {
                'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
                'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
            }[optimizer_type]
            optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})
            optimizer.add_param_group({'params': pg2})

            lr_scheduler = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, local_epochs)

            train_dataset = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=local_epochs, train=True)
            loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=True, drop_last=True,
                                collate_fn=yolo_dataset_collate,
                                worker_init_fn=partial(worker_init_fn, rank=client_id, seed=seed))

            for epoch in range(local_epochs):
                set_optimizer_lr(optimizer, lr_scheduler, epoch)
                fit_one_epoch(local_model, local_model, ema, yolo_loss,
                              loss_history,
                              EvalCallback(local_model, input_shape,
                                           anchors, anchors_mask,
                                           class_names, num_classes,
                                           val_lines1, val_lines2, client_save_dir,
                                           Cuda, eval_flag=eval_flag,
                                           period=eval_period),
                              optimizer, epoch,
                              len(loader), len(loader), loader,
                              local_epochs, Cuda, fp16,
                              None,
                              save_period=save_period,
                              save_dir=client_save_dir,
                              local_rank=local_rank, round_idx=rnd+1)

            local_states.append(local_model.state_dict())
            data_sizes.append(len(train_lines))
            torch.cuda.empty_cache()

        print("-> Aggregating local models with Weighted FedAvg + Momentum...")
        new_state = weighted_average_weights(local_states, data_sizes)

        if momentum_state is None:
            momentum_state = new_state
        else:
            for k in new_state:
                momentum_state[k] = fed_momentum * momentum_state[k] + (1 - fed_momentum) * new_state[k]

        global_model.load_state_dict(momentum_state)

        if local_rank == 0:
            agg_save_path = os.path.join(save_dir, f"Global_round_{rnd+1}_weights.pth")
            torch.save(global_model.state_dict(), agg_save_path)
            print(f"Saved aggregated global model to {agg_save_path}")

        global_model.eval()
        global_eval_cb = EvalCallback(global_model, input_shape,
                                      anchors, anchors_mask,
                                      class_names, num_classes,
                                      val_lines1, val_lines2, save_dir,
                                      Cuda, eval_flag=True,
                                      period=1)
        global_eval_cb.on_epoch_end(rnd, global_model)

        if local_rank == 0 and fed_loss_history is not None:
            fed_loss_history.append_loss(rnd+1, 0, 0, 0)

    print("\nFederated training completed.")
