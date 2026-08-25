'''
Thanks for the code release from WangXun from: https://github.com/bnu-wangxun/Deep_Metric
if use this code, please consider the paper:

@inproceedings{wang2019multi,
title={Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},
author={Wang, Xun and Han, Xintong and Huang, Weilin and Dong, Dengke and Scott, Matthew R},
booktitle={CVPR},
year={2019}
}

@article{chen2021feature,
title={Feature Estimations based Correlation Distillation for Incremental Image Retrieval},
author={Wei Chen and Yu Liu and Nan Pu and Weiping Wang and Li Liu and Lew Michael S},
journal={IEEE Transactions on Multimedia},
year={2021},
}
'''

# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch
import torch.utils.data
from torch.backends import cudnn
import models
import losses
from utils import (
    FastRandomIdentitySampler, mkdir_if_missing, logging, display,
    unwrap_model, get_loader_kwargs, wrap_model, get_device, get_num_gpus,
    get_device_ids, model_state_dict, get_model_attribute
)
from utils.serialization import save_checkpoint
from trainer import train
import torch.nn as nn
from evaluations import (
    extract_features, pairwise_similarity, evaluate_incremental_retrieval,
    log_results_csv, log_history_json
)
from Model2Feature import Model2Feature

import DataSet
import os.path as osp
from losses.L2_norm import L2Norm
from losses.Similarity_preserving_loss import *

cudnn.benchmark = True


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def create_model_and_optimizer(args, device, num_classes, resume_path=None):
    """Create model, frozen model, and optimizer"""
    model = models.create(args.net, pretrained=True, dim=args.dim)
    model_frozen = models.create(args.net, pretrained=True, dim=args.dim)
    
    if resume_path is not None:
        print('Loading model from {}'.format(resume_path))
        chk_pt = torch.load(resume_path, map_location=device)
        weight = chk_pt['state_dict']
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        model_dict_frozen = model_frozen.state_dict()
        pretrained_dict_frozen = {k: v for k, v in weight.items() if k in model_dict_frozen}
        model_dict_frozen.update(pretrained_dict_frozen)
        model_frozen.load_state_dict(model_dict_frozen)
        model_frozen.eval()
    
    model = wrap_model(model, args.gpu_ids)
    model_frozen = wrap_model(model_frozen, args.gpu_ids)
    
    if args.freeze_BN:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40 * '#', 'BatchNorm NOT frozen')
    
    unwrapped_model = unwrap_model(model)
    new_param_ids_fc_layer = set(map(id, get_model_attribute(unwrapped_model, 'fc_layer').parameters()))
    new_params_fc = [p for p in unwrapped_model.parameters() if id(p) in new_param_ids_fc_layer]
    base_params = [p for p in unwrapped_model.parameters() if id(p) not in new_param_ids_fc_layer]
    
    for p in unwrap_model(model_frozen).parameters():
        p.requires_grad = False
    
    param_groups = [
        {'params': base_params, 'lr_mult': 0.1},
        {'params': new_params_fc, 'lr_mult': 1.0}
    ]
    
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    criterion_loss = losses.create(args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).to(device)
    CE_loss = nn.CrossEntropyLoss().to(device)
    l2_loss = L2Norm().to(device)
    similarity_loss = Similarity_preserving().to(device)
    
    criterion = [criterion_loss, CE_loss, l2_loss, similarity_loss]
    
    return model, model_frozen, optimizer, criterion


def create_data_loaders(args, task_id):
    """Create data loaders for a specific task"""
    data = DataSet.create(
        args.data, 
        ratio=args.ratio, 
        width=args.width, 
        origin_width=args.origin_width, 
        root=args.data_root,
        task_id=task_id,
        max_tasks=args.max_tasks
    )
    
    loader_kwargs = get_loader_kwargs(
        batch_size=args.batch_size,
        num_workers=args.nThreads,
        gpu_ids=args.gpu_ids,
        drop_last=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        data.train,
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        **loader_kwargs
    )
    
    val_loader = torch.utils.data.DataLoader(
        data.val,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    gallery_loader = torch.utils.data.DataLoader(
        data.gallery,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    return data, train_loader, val_loader, gallery_loader


def evaluate_model(model, data, args, task_id, seen_classes, task_maps_history):
    """Evaluate model on current task"""
    device = get_device(args.gpu_ids)
    
    gallery_feature, gallery_labels, query_feature, query_labels, _, _ = Model2Feature(
        data=args.data, 
        root=args.data_root, 
        width=args.width, 
        net=args.net, 
        checkpoint={'state_dict': model_state_dict(model)}, 
        dim=args.dim, 
        batch_size=args.batch_size, 
        nThreads=args.nThreads
    )
    
    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    
    if args.gallery_eq_query:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))
    
    results = evaluate_incremental_retrieval(
        sim_mat, query_labels, gallery_labels, seen_classes, task_id, task_maps_history
    )
    
    return results


def run_incremental_training(args):
    """Main incremental training loop"""
    device = get_device(args.gpu_ids)
    num_gpus = get_num_gpus(args.gpu_ids)
    print(f"Using {num_gpus} GPU(s): {get_device_ids(args.gpu_ids)}")
    
    if args.data == 'ip102':
        max_tasks = args.max_tasks
    else:
        max_tasks = 1
    
    task_maps_history = []
    all_seen_classes = set()
    
    model = None
    model_frozen = None
    optimizer = None
    criterion = None
    
    for task_id in range(max_tasks):
        print(f"\n{'='*60}")
        print(f"Starting Task {task_id + 1}/{max_tasks}")
        print(f"{'='*60}")
        
        if args.data == 'ip102':
            temp_data = DataSet.create(
                args.data, 
                ratio=args.ratio, 
                width=args.width, 
                origin_width=args.origin_width, 
                root=args.data_root,
                task_id=task_id,
                max_tasks=args.max_tasks
            )
            task_classes = set(temp_data.train.classes)
            all_seen_classes.update(task_classes)
            print(f"Task {task_id} classes: {sorted(task_classes)}")
            print(f"Total seen classes: {len(all_seen_classes)}")
        
        if task_id == 0:
            resume_path = args.resume
            incremental_flag = False
        else:
            resume_path = osp.join(args.save_dir, f'ckp_task{task_id}_best.pth.tar')
            incremental_flag = True
        
        args.Incremental_flag = incremental_flag
        
        model, model_frozen, optimizer, criterion = create_model_and_optimizer(
            args, device, len(all_seen_classes), resume_path
        )
        
        data, train_loader, val_loader, gallery_loader = create_data_loaders(args, task_id)
        
        print(f'Training on task {task_id} with {len(data.train)} samples, {len(data.train.classes)} classes')
        
        best_map = 0.0
        start_epoch = 0
        
        if resume_path and osp.exists(resume_path):
            chk_pt = torch.load(resume_path, map_location=device)
            start_epoch = chk_pt.get('epoch', 0)
        
        for epoch in range(start_epoch, args.epochs):
            accuracy = train(
                epoch=epoch, 
                model=[model, model_frozen], 
                criterion=criterion, 
                optimizer=optimizer, 
                train_loader=train_loader, 
                args=args
            )
            
            if (epoch + 1) % args.save_step == 0 or epoch == 0:
                results = evaluate_model(model, data, args, task_id, all_seen_classes, task_maps_history)
                current_map = results.get('mAP', 0.0)
                task_maps_history.append(current_map)
                
                print(f"Task {task_id}, Epoch {epoch + 1}: mAP={current_map:.4f}")
                for k in [1, 5, 10]:
                    print(f"  R@{k}={results.get(f'R@{k}', 0):.4f}", end=' ')
                print()
                
                if current_map > best_map:
                    best_map = current_map
                    save_checkpoint({
                        'state_dict': model_state_dict(model),
                        'epoch': epoch + 1,
                        'task_id': task_id,
                        'seen_classes': list(all_seen_classes),
                    }, True, fpath=osp.join(args.save_dir, f'ckp_task{task_id}_best.pth.tar'))
                
                save_checkpoint({
                    'state_dict': model_state_dict(model),
                    'epoch': epoch + 1,
                    'task_id': task_id,
                    'seen_classes': list(all_seen_classes),
                }, False, fpath=osp.join(args.save_dir, f'ckp_task{task_id}_ep{epoch + 1}.pth.tar'))
                
                log_results_csv(results, task_id, len(all_seen_classes), 
                                csv_path=osp.join(args.save_dir, 'results.csv'))
                log_history_json(results, task_id, len(all_seen_classes), 
                                 json_path=osp.join(args.save_dir, 'history.json'))
        
        final_results = evaluate_model(model, data, args, task_id, all_seen_classes, task_maps_history)
        task_maps_history.append(final_results.get('mAP', 0.0))
        
        log_results_csv(final_results, task_id, len(all_seen_classes), 
                        csv_path=osp.join(args.save_dir, 'results.csv'))
        log_history_json(final_results, task_id, len(all_seen_classes), 
                         json_path=osp.join(args.save_dir, 'history.json'))
        
        print(f"\nTask {task_id} completed. Best mAP: {best_map:.4f}")
    
    print("\nTraining completed!")
    print(f"Results saved to {osp.join(args.save_dir, 'results.csv')}")
    print(f"History saved to {osp.join(args.save_dir, 'history.json')}")


def main(args):
    mkdir_if_missing(args.save_dir)
    sys.stdout = logging.Logger(os.path.join(args.save_dir, 'log.txt'))
    display(args)
    
    if args.max_tasks > 1 or args.data == 'ip102':
        run_incremental_training(args)
    else:
        run_single_task_training(args)


def run_single_task_training(args):
    """Original single-task training for backward compatibility"""
    device = get_device(args.gpu_ids)
    
    model = models.create(args.net, pretrained=True, dim=args.dim)
    model_frozen = models.create(args.net, pretrained=True, dim=args.dim)
    
    if args.resume is not None:
        print('load model from {}'.format(args.resume))
        chk_pt = torch.load(args.resume, map_location=device)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        model_dict_frozen = model_frozen.state_dict()
        pretrained_dict_frozen = {k: v for k, v in weight.items() if k in model_dict_frozen}
        model_dict_frozen.update(pretrained_dict_frozen)
        model_frozen.load_state_dict(model_dict_frozen)
        model_frozen.eval()
    else:
        start = 0
    
    model = wrap_model(model, args.gpu_ids)
    model_frozen = wrap_model(model_frozen, args.gpu_ids)
    
    if args.freeze_BN:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40 * '#', 'BatchNorm NOT frozen')
    
    unwrapped_model = unwrap_model(model)
    new_param_ids_fc_layer = set(map(id, get_model_attribute(unwrapped_model, 'fc_layer').parameters()))
    new_params_fc = [p for p in unwrapped_model.parameters() if id(p) in new_param_ids_fc_layer]
    base_params = [p for p in unwrapped_model.parameters() if id(p) not in new_param_ids_fc_layer]
    
    for p in unwrap_model(model_frozen).parameters():
        p.requires_grad = False
    
    param_groups = [
        {'params': base_params, 'lr_mult': 0.1},
        {'params': new_params_fc, 'lr_mult': 1.0}
    ]
    
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    criterion_loss = losses.create(args.loss, margin=args.margin, alpha=args.alpha, base=args.loss_base).to(device)
    CE_loss = nn.CrossEntropyLoss().to(device)
    l2_loss = L2Norm().to(device)
    similarity_loss = Similarity_preserving().to(device)
    criterion = [criterion_loss, CE_loss, l2_loss, similarity_loss]
    
    data = DataSet.create(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root)
    
    loader_kwargs = get_loader_kwargs(args.batch_size, args.nThreads, args.gpu_ids, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        data.train, 
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        **loader_kwargs
    )
    
    best_accuracy = 0
    model_list = [model, model_frozen]
    
    if args.Incremental_flag == False:
        print("###################### This is non-incremental learning! ########################")
    elif args.Incremental_flag == True:
        print("######################### This is incremental learning! #########################")
    else:
        raise NotImplementedError()
    
    for epoch in range(start, args.epochs):
        accuracy = train(epoch=epoch, model=model_list, criterion=criterion, optimizer=optimizer, train_loader=train_loader, args=args)
        
        if (epoch + 1) % args.save_step == 0 or epoch == 0:
            if get_num_gpus(args.gpu_ids) > 0:
                state_dict = model_state_dict(model)
            else:
                state_dict = model.state_dict()
            
            is_best = accuracy > best_accuracy
            best_accuracy = max(accuracy, best_accuracy)
            
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch + 1),
            }, is_best, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental Fine-grained Image Retrieval')
    
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--batch_size', '-b', default=80, type=int, help='mini-batch size')
    parser.add_argument('--num_instances', default=5, type=int, help='number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, help='dimension of embedding space')
    parser.add_argument('--width', default=224, type=int, help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int, help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float, help='random crop ratio for train data')
    
    parser.add_argument('--alpha', default=30, type=int, help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=1, type=float, help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.5, type=float, help='margin in loss function')
    parser.add_argument('--init', default='random', help='the initialization way of FC layer')
    
    parser.add_argument('--Incremental_flag', default=False, type=bool, help='incremental learning or not')
    parser.add_argument('--data', default='cub', help='name of Data Set (cub, dog, ip102)')
    parser.add_argument('--freeze_BN', default=True, type=bool, help='Freeze BN if True')
    parser.add_argument('--data_root', type=str, default='data', help='path to Data Set')
    
    parser.add_argument('--net', default='BN_Inception')
    parser.add_argument('--loss', default='HardMining', help='loss for training network')
    parser.add_argument('--epochs', default=2300, type=int, help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, help='number of epochs to save model')
    
    parser.add_argument('--resume', '-r', default=None, help='resume from checkpoint')
    parser.add_argument('--resume_pre_step_2', default=None, help='resume for step 2')
    
    parser.add_argument('--print_freq', default=6, type=int, help='display frequency of training')
    parser.add_argument('--save_dir', default='ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80')
    parser.add_argument('--nThreads', '-j', default=16, type=int, help='number of data loading threads')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--loss_base', type=float, default=0.75)
    
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs (e.g., 0,1,2,3 or "auto")')
    parser.add_argument('--max_tasks', type=int, default=4, help='maximum number of incremental tasks (for IP102)')
    parser.add_argument('--gallery_eq_query', default=True, type=bool, help='gallery equals query')
    
    args = parser.parse_args()
    
    if args.gpu_ids == 'auto':
        args.gpu_ids = ','.join(str(i) for i in range(torch.cuda.device_count()))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids if isinstance(args.gpu_ids, str) else ','.join(map(str, args.gpu_ids))
    
    print('Arguments:', args)
    main(args)