from __future__ import absolute_import, print_function
"""
IP102 dataset for PyTorch - Incremental Image Retrieval
25 classes split into 4 tasks: 7/6/6/6
"""
import torch
import torch.utils.data as data
from PIL import Image
import os
import json
from collections import defaultdict
import numpy as np
from DataSet import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


def Generate_transform_Dict(origin_width=256, width=227, ratio=0.16):
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0/255, 1.0/255, 1.0/255])

    transform_dict = {}

    transform_dict['rand-crop'] = \
    transforms.Compose([
                transforms.CovertBGR(),
                transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['center-crop'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_dict['resize'] = \
    transforms.Compose([
                    transforms.CovertBGR(),
                    transforms.Resize((width)),
                    transforms.ToTensor(),
                    normalize,
                ])
    return transform_dict


class IP102Data(data.Dataset):
    def __init__(self, root=None, split='train', transform=None, loader=default_loader,
                 task_id=0, use_filtered_classes=True):
        """
        Args:
            root: root directory of dataset
            split: 'train', 'val', or 'test'
            transform: image transform
            loader: image loader
            task_id: which incremental task (0-3), -1 means all 25 classes
            use_filtered_classes: whether to use only 25 filtered classes
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.loader = loader
        self.task_id = task_id
        self.use_filtered_classes = use_filtered_classes
        
        self.filtered_class_ids = self._load_filtered_classes()
        self.class_id_to_name = self._load_class_names()
        self.task_class_splits = self._get_task_splits()
        
        self.images, self.labels, self.image_ids = self._load_data()
        self.classes = sorted(list(set(self.labels)))
        
        self.Index = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.Index[label].append(i)

    def _find_data_root(self):
        """Auto-find data root directory (supports local, Kaggle, env var)"""
        if self.root and os.path.exists(self.root):
            return self.root
        
        search_paths = [
            os.environ.get('IP102_DATA_ROOT', ''),
            '/kaggle/input/datasets/nta212/ip102-for-object-detection',  # User's specific Kaggle path
            '/kaggle/input/ip102-dataset',
            '/kaggle/input/ip102',
            '/kaggle/input/IP102 dataset',
            'D:/Sau_Benh_object/retrieval-img/IP102 dataset',
            './IP102 dataset',
            '../IP102 dataset',
            '../../IP102 dataset',
        ]
        
        for path in search_paths:
            if path and os.path.exists(path):
                json_files = ['train.json', 'val.json', 'test.json']
                if all(os.path.exists(os.path.join(path, f)) for f in json_files):
                    return path
                voc_path = os.path.join(path, 'VOC2007', 'VOC2007', 'JPEGImages')
                if os.path.exists(voc_path):
                    return path
        
        raise FileNotFoundError(
            "IP102 dataset not found. Set IP102_DATA_ROOT env var or place dataset in expected locations."
        )

    def _load_filtered_classes(self):
        """Load 25 filtered class IDs from filtered_class.txt"""
        data_root = self._find_data_root()
        filtered_path = os.path.join(data_root, 'filtered_class.txt')
        if not os.path.exists(filtered_path):
            raise FileNotFoundError(f"filtered_class.txt not found at {filtered_path}")
        
        with open(filtered_path, 'r') as f:
            class_ids = [int(line.strip()) for line in f if line.strip()]
        return class_ids

    def _load_class_names(self):
        """Load class ID to name mapping from classes.txt"""
        data_root = self._find_data_root()
        classes_path = os.path.join(data_root, 'classes.txt')
        if not os.path.exists(classes_path):
            return {}
        
        class_map = {}
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_map[int(parts[0])] = parts[1].strip()
        return class_map

    def _get_task_splits(self):
        """Split 25 classes into 4 tasks: 7/6/6/6"""
        return [
            self.filtered_class_ids[0:7],    # Task 0: 7 classes
            self.filtered_class_ids[7:13],   # Task 1: 6 classes
            self.filtered_class_ids[13:19],  # Task 2: 6 classes
            self.filtered_class_ids[19:25],  # Task 3: 6 classes
        ]

    def _get_classes_for_task(self, task_id):
        """Get class IDs for a specific task or all tasks up to task_id"""
        if task_id == -1:
            return self.filtered_class_ids
        if task_id < 0 or task_id >= len(self.task_class_splits):
            raise ValueError(f"Invalid task_id: {task_id}. Must be 0-3 or -1 for all.")
        
        classes = []
        for i in range(task_id + 1):
            classes.extend(self.task_class_splits[i])
        return classes

    def _load_data(self):
        data_root = self._find_data_root()
        json_file = os.path.join(data_root, f'{self.split}.json')
        
        if not os.path.exists(json_file):
            if self.split == 'val':
                json_file = os.path.join(data_root, 'test.json')
                if not os.path.exists(json_file):
                    raise FileNotFoundError(f"Neither val.json nor test.json found at {data_root}")
            else:
                raise FileNotFoundError(f"{self.split}.json not found at {json_file}")
        
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        image_id_to_info = {img['id']: img for img in coco_data['images']}
        
        valid_class_ids = set(self._get_classes_for_task(self.task_id)) if self.use_filtered_classes else None
        
        images = []
        labels = []
        image_ids = []
        
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            if valid_class_ids is not None and cat_id not in valid_class_ids:
                continue
            
            img_id = ann['image_id']
            if img_id not in image_id_to_info:
                continue
            
            img_info = image_id_to_info[img_id]
            img_path = self._find_image_path(data_root, img_info['file_name'])
            
            if img_path and os.path.exists(img_path):
                images.append(img_path)
                labels.append(cat_id)
                image_ids.append(img_id)
        
        if len(images) == 0:
            raise RuntimeError(f"No images found for split={self.split}, task_id={self.task_id}")
        
        print(f"IP102 {self.split}: loaded {len(images)} images, {len(set(labels))} classes")
        return images, labels, image_ids

    def _find_image_path(self, data_root, file_name):
        """Find image file in various possible locations"""
        possible_paths = [
            os.path.join(data_root, 'VOC2007', 'VOC2007', 'JPEGImages', file_name),
            os.path.join(data_root, 'JPEGImages', file_name),
            os.path.join(data_root, 'images', file_name),
            os.path.join(data_root, file_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def get_class_name(self, class_id):
        """Get class name for a class ID"""
        return self.class_id_to_name.get(class_id, str(class_id))

    def get_task_info(self):
        """Get information about task splits"""
        info = {}
        for i, split in enumerate(self.task_class_splits):
            info[f'task_{i}'] = {
                'classes': split,
                'class_names': [self.get_class_name(c) for c in split],
                'num_classes': len(split)
            }
        info['all'] = {
            'classes': self.filtered_class_ids,
            'class_names': [self.get_class_name(c) for c in self.filtered_class_ids],
            'num_classes': len(self.filtered_class_ids)
        }
        return info

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        img_name = img_path
        
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, img_name

    def __len__(self):
        return len(self.images)


class IP102:
    def __init__(self, width=227, origin_width=256, ratio=0.16, root=None, transform=None,
                 task_id=0, max_tasks=4, new_label_start_point=0):
        print(f'width: \t {width}')
        print(f'task_id: \t {task_id}')
        print(f'max_tasks: \t {max_tasks}')
        
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = "data/IP102"
        
        self.task_id = task_id
        self.max_tasks = max_tasks
        self.root = root
        
        self.train = IP102Data(root, split='train', transform=transform_Dict['rand-crop'],
                               task_id=task_id, use_filtered_classes=True)
        self.gallery = IP102Data(root, split='test', transform=transform_Dict['center-crop'],
                                 task_id=task_id, use_filtered_classes=True)
        
        val_data = IP102Data(root, split='val', transform=transform_Dict['center-crop'],
                             task_id=task_id, use_filtered_classes=True)
        self.val = val_data if len(val_data) > 0 else self.gallery
        
        self.num_classes = len(self.train.classes)
        print(f'Number of classes in task {task_id}: {self.num_classes}')
        print(f'Train samples: {len(self.train)}, Gallery samples: {len(self.gallery)}, Val samples: {len(self.val)}')


def create_ip102_dataset(width=227, origin_width=256, ratio=0.16, root=None, transform=None,
                         task_id=0, max_tasks=4, new_label_start_point=0):
    return IP102(width=width, origin_width=origin_width, ratio=ratio, root=root, transform=transform,
                 task_id=task_id, max_tasks=max_tasks, new_label_start_point=new_label_start_point)