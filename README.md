# Deep-Incremental-Image-Retrieval for IP102

PyTorch implementation for **Feature Estimations based Correlation Distillation for Incremental Image Retrieval** (IEEE Transactions on Multimedia, 2021).

Extended to support **IP102 dataset** with incremental learning (4 tasks: 7/6/6/6 classes).

## Dependencies

- Python 3.8+
- PyTorch 1.4.0+ (tested with 1.10+)
- torchvision
- numpy
- scikit-learn
- pillow
- pandas (for results viewing)

Install: `pip install torch torchvision numpy scikit-learn pillow pandas`

## Datasets Supported

| Dataset | Classes | Tasks | Split |
|---------|---------|-------|-------|
| **IP102** | 25 | 4 | 7/6/6/6 |
| CUB-200-2011 | 200 | 2 | 100/100 |
| Stanford-Dogs-120 | 120 | 2 | 60/60 |

### IP102 Dataset Structure

The IP102 dataset uses COCO format with 25 filtered classes from 102 total classes.

**Required files in dataset root:**
```
ip102-for-object-detection/
├── train.json          # COCO format annotations
├── val.json            # COCO format annotations  
├── test.json           # COCO format annotations
├── filtered_class.txt  # 25 class IDs (one per line)
├── classes.txt         # Class ID to name mapping (102 lines)
└── VOC2007/
    └── VOC2007/
        └── JPEGImages/ # Image files
```

**Kaggle paths:**
- Dataset: `/kaggle/input/datasets/nta212/ip102-for-object-detection`
- Pretrained weight: `/kaggle/input/models/nhannguyen5578/deep-incremental-image-retrieval-pretrain/pytorch/default/1/bn_inception-52deb4733.pth`

## Pretrained Backbone

**BN-Inception** pretrained on ImageNet.

Download from: [bn_inception-52deb4733.pth](https://drive.google.com/file/d/1qDBfquYrfM9Msl2q57jxzl9w0y7qwnn0/view?usp=sharing)

Place at: `models/bn_inception-52deb4733.pth` or use Kaggle path above.

## Training

### Quick Start (Local)

```bash
# Single task (CUB/Dog) - backward compatible
python train.py --data cub --epochs 1500 --batch_size 80 --gpu_ids 0

# IP102 Incremental (4 tasks)
python train.py --data ip102 --max_tasks 4 --epochs 100 --batch_size 32 --gpu_ids 0,1 --data_root /path/to/ip102
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Dataset name: `cub`, `dog`, `ip102` | `cub` |
| `--max_tasks` | Number of incremental tasks (IP102) | `4` |
| `--epochs` | Epochs per task | `2300` |
| `--batch_size` | Batch size per GPU | `80` |
| `--lr` | Learning rate | `1e-5` |
| `--dim` | Embedding dimension | `512` |
| `--gpu_ids` | GPU IDs (e.g., `0,1` or `auto`) | `0` |
| `--save_dir` | Checkpoint directory | `ckps/...` |
| `--resume` | Pretrained weight path | `None` |
| `--freeze_BN` | Freeze BatchNorm | `True` |

### Kaggle Notebook

Open `kaggle_notebook.ipynb` on Kaggle:
1. Add dataset input: `nta212/ip102-for-object-detection`
2. Add model input: `nhannguyen5578/deep-incremental-image-retrieval-pretrain`
3. Set environment variable `IP102_CODE_REPO` to `https://github.com/nta2112/Deep-Incremental-Image-Retrieval-for-IP102.git`
4. Run all cells

```python
# Quick test (2 tasks, 1 epoch each)
results_csv, history_json = run_train(
    model_name='BN_Inception',
    max_tasks=2,
    epochs=1,
    batch_size=16,
    gpu_ids='auto',
    pretrained_path=PRETRAINED_WEIGHT_PATH
)

# Full training (4 tasks, 100 epochs each)
results_csv, history_json = run_train(
    model_name='BN_Inception',
    max_tasks=4,
    epochs=100,
    batch_size=32,
    gpu_ids='auto',
    pretrained_path=PRETRAINED_WEIGHT_PATH
)
```

## Evaluation Metrics

After each task, the following metrics are logged to `results.csv` and `history.json`:

### Retrieval
- **R@1, R@5, R@10**: Recall at K
- **mAP**: Mean Average Precision (macro)

### Open-World (OOD Detection)
- **AUROC**: Area Under ROC Curve
- **FPR@TPR95**: False Positive Rate at 95% True Positive Rate
- **Recall@1_Seen**: Recall@1 on seen classes
- **Recall@1_Unseen**: Recall@1 on unseen classes (None when all classes seen)

### Lifelong Learning
- **Plasticity**: Average mAP across tasks
- **Forgetting**: Average performance drop on previous tasks
- **Overall**: Plasticity - Forgetting

## Results Format

**results.csv header:**
```
task,numclass,cnn_top1,nme_top1,R@1,R@5,R@10,mAP,AUROC,FPR95,Plasticity,Forgetting,Overall
```

## Multi-GPU Support

The code automatically handles multi-GPU training:
- Batch size adjusted to be divisible by number of GPUs
- `drop_last=True` for DataLoader
- Model unwrapping helpers: `unwrap_model()`, `get_model_attribute()`

## Project Structure

```
Deep-Incremental-Image-Retrieval/
├── train.py                 # Main training script (incremental loop)
├── test.py                  # Testing script
├── trainer.py               # Training step implementation
├── Model2Feature.py         # Feature extraction
├── kaggle_notebook.ipynb    # Kaggle notebook
├── DataSet/
│   ├── __init__.py          # Dataset factory
│   ├── IP102.py             # IP102 dataset (NEW)
│   ├── transforms.py        # Custom transforms (NEW)
│   ├── CUB200.py            # CUB-200 dataset
│   └── Stanford_dog.py      # Stanford-Dogs dataset
├── models/
│   ├── __init__.py
│   └── BN_Inception.py      # BN-Inception backbone
├── losses/
│   ├── __init__.py
│   ├── HardMining.py        # Multi-Similarity loss
│   ├── L2_norm.py
│   └── Similarity_preserving_loss.py
├── evaluations/
│   ├── __init__.py
│   ├── metrics.py           # All metrics (NEW)
│   ├── recall_at_k.py
│   ├── extract_featrure.py
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── multi_gpu.py         # Multi-GPU helpers (NEW)
│   ├── sampler.py
│   ├── serialization.py
│   └── ...
└── ckps/                    # Checkpoints (auto-created)
```

## Citation

```bibtex
@article{chen2021feature,
  title={Feature Estimations based Correlation Distillation for Incremental Image Retrieval},
  author={Wei Chen and Yu Liu and Nan Pu and Weiping Wang and Li Liu and Lew Michael S},
  journal={IEEE Transactions on Multimedia},
  year={2021},
}

@inproceedings{wang2019multi,
  title={Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},
  author={Wang, Xun and Han, Xintong and Huang, Weilin and Dong, Dengke and Scott, Matthew R},
  booktitle={CVPR},
  year={2019}
}
```

## Acknowledgments

- Original code from [WangXun/Deep_Metric](https://github.com/bnu-wangxun/Deep_Metric)
- IP102 dataset from [IP102](https://github.com/xpzhu/IP102)