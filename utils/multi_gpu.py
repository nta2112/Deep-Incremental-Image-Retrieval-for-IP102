"""
Multi-GPU safety helpers for DataParallel training.
"""
import torch
import torch.nn as nn
import os


def unwrap_model(model):
    """
    Unwrap model from DataParallel if wrapped.
    Returns the underlying module.
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def get_device_ids(gpu_ids=None):
    """
    Get list of GPU device IDs.
    If gpu_ids is None, auto-detect available GPUs.
    If gpu_ids is string like '0,1', parse it.
    If gpu_ids is list, use as is.
    """
    if gpu_ids is None:
        gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if isinstance(gpu_ids, str):
        if gpu_ids.strip() == '':
            return []
        return [int(x.strip()) for x in gpu_ids.split(',') if x.strip()]
    
    if isinstance(gpu_ids, (list, tuple)):
        return list(gpu_ids)
    
    return []


def get_num_gpus(gpu_ids=None):
    """Get number of GPUs to use"""
    return len(get_device_ids(gpu_ids))


def get_loader_kwargs(batch_size, num_workers, gpu_ids=None, drop_last=True):
    """
    Get DataLoader kwargs adjusted for multi-GPU.
    Ensures batch_size is divisible by num_gpus.
    """
    num_gpus = get_num_gpus(gpu_ids)
    
    if num_gpus > 1:
        if batch_size % num_gpus != 0:
            adjusted_batch_size = (batch_size // num_gpus) * num_gpus
            if adjusted_batch_size == 0:
                adjusted_batch_size = num_gpus
            print(f"Adjusting batch_size from {batch_size} to {adjusted_batch_size} "
                  f"to be divisible by {num_gpus} GPUs")
            batch_size = adjusted_batch_size
        
        return {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': drop_last,
        }
    else:
        return {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': drop_last,
        }


def wrap_model(model, gpu_ids=None):
    """
    Wrap model with DataParallel if multiple GPUs available.
    """
    device_ids = get_device_ids(gpu_ids)
    
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Model wrapped with DataParallel on GPUs: {device_ids}")
    elif len(device_ids) == 1:
        model = model.cuda(device_ids[0])
        print(f"Model moved to GPU: {device_ids[0]}")
    else:
        print("No GPU available, using CPU")
    
    return model


def get_device(gpu_ids=None):
    """Get primary device for model operations"""
    device_ids = get_device_ids(gpu_ids)
    if len(device_ids) > 0:
        return torch.device(f'cuda:{device_ids[0]}')
    return torch.device('cpu')


def synchronize_batchnorm(model):
    """
    Synchronize BatchNorm statistics across GPUs (for DataParallel).
    This is automatically handled by DataParallel in recent PyTorch versions.
    """
    pass


def model_state_dict(model):
    """Get state_dict from model, handling DataParallel"""
    return unwrap_model(model).state_dict()


def load_model_state(model, state_dict, strict=True):
    """Load state_dict into model, handling DataParallel"""
    return unwrap_model(model).load_state_dict(state_dict, strict=strict)


def get_model_attribute(model, attr_name):
    """
    Safely get attribute from model, handling DataParallel.
    Usage: get_model_attribute(model, 'fc_layer') instead of model.fc_layer or model.module.fc_layer
    """
    return getattr(unwrap_model(model), attr_name)


def set_model_attribute(model, attr_name, value):
    """
    Safely set attribute on model, handling DataParallel.
    """
    setattr(unwrap_model(model), attr_name, value)


def is_data_parallel(model):
    """Check if model is wrapped in DataParallel"""
    return isinstance(model, nn.DataParallel)


def get_module(model):
    """Alias for unwrap_model"""
    return unwrap_model(model)