# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
from utils import wrap_model, get_loader_kwargs, get_device
cudnn.benchmark = True


def Model2Feature(data, net, checkpoint, dim=512, width=224, root=None, Retrieval_visualization=False, nThreads=16, batch_size=100, pool_feature=False, task_id=0, max_tasks=1, gpu_ids='0', **kargs):
    dataset_name = data
    model = models.create(net, dim=dim, pretrained=False)

    resume = checkpoint
    net_dict = model.state_dict()
    weights = resume['state_dict']
    pretrained_dict = {k: v for k, v in weights.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)

    model = wrap_model(model, gpu_ids)
    device = get_device(gpu_ids)
    model = model.to(device)
    
    data = DataSet.create(data, width=width, root=root, task_id=task_id, max_tasks=max_tasks)
    
    loader_kwargs = get_loader_kwargs(batch_size, nThreads, gpu_ids, drop_last=False)
    
    if dataset_name in ['shop', 'jd_test']:
        gallery_loader = torch.utils.data.DataLoader(data.gallery, **loader_kwargs)
        query_loader = torch.utils.data.DataLoader(data.query, **loader_kwargs)

        gallery_feature, gallery_labels, img_name = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        query_feature, query_labels, img_name = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    else:
        data_loader = torch.utils.data.DataLoader(data.gallery, **loader_kwargs)

        if Retrieval_visualization:
            data_loader_shuffled = torch.utils.data.DataLoader(
                data.gallery, shuffle=True, **loader_kwargs)
        else:
            data_loader_shuffled = torch.utils.data.DataLoader(
                data.gallery, shuffle=False, **loader_kwargs)

        features, labels, img_name = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        features_shuffled, labels_shuffled, img_name_shuffled = extract_features(model, data_loader_shuffled, print_freq=1e5, metric=None, pool_feature=pool_feature)

        gallery_feature, gallery_labels = features, labels
        query_feature, query_labels = features_shuffled, labels_shuffled

    return gallery_feature, gallery_labels, query_feature, query_labels, img_name, img_name_shuffled

