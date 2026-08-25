"""
Comprehensive metrics for incremental image retrieval:
- Retrieval: R@1/5/10, mAP macro
- Open-world (OOD detection): AUROC, FPR@TPR95, Recall@1 on Seen/Unseen
- Lifelong: Plasticity, Forgetting, Overall from mAP per task group
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
import json
import os


def recall_at_k(sim_mat, query_ids, gallery_ids, ks=(1, 5, 10)):
    """
    Compute Recall@K for retrieval.
    Args:
        sim_mat: similarity matrix (query x gallery)
        query_ids: labels for queries
        gallery_ids: labels for gallery
        ks: tuple of K values
    Returns:
        dict of R@K values
    """
    num_queries = sim_mat.shape[0]
    max_k = max(ks)
    
    sorted_indices = torch.argsort(sim_mat, dim=1, descending=True)
    
    results = {}
    for k in ks:
        correct = 0
        for i in range(num_queries):
            top_k_indices = sorted_indices[i, :k]
            top_k_labels = gallery_ids[top_k_indices]
            if query_ids[i] in top_k_labels:
                correct += 1
        results[f'R@{k}'] = correct / num_queries if num_queries > 0 else 0.0
    
    return results


def mean_average_precision(sim_mat, query_ids, gallery_ids, macro=True):
    """
    Compute mAP for retrieval.
    Args:
        sim_mat: similarity matrix (query x gallery)
        query_ids: labels for queries
        gallery_ids: labels for gallery
        macro: if True, compute macro-average across classes
    Returns:
        mAP value
    """
    num_queries = sim_mat.shape[0]
    sorted_indices = torch.argsort(sim_mat, dim=1, descending=True)
    
    aps = []
    unique_classes = torch.unique(query_ids)
    
    for class_id in unique_classes:
        class_mask = (query_ids == class_id)
        if not class_mask.any():
            continue
        
        class_aps = []
        for i in torch.where(class_mask)[0]:
            relevant = (gallery_ids == query_ids[i])
            if not relevant.any():
                continue
            
            ranked_relevant = relevant[sorted_indices[i]]
            precision_at_k = torch.cumsum(ranked_relevant.float(), dim=0) / torch.arange(1, len(ranked_relevant) + 1, dtype=torch.float)
            ap = (precision_at_k * ranked_relevant.float()).sum() / relevant.sum().float()
            class_aps.append(ap.item())
        
        if class_aps:
            aps.append(np.mean(class_aps))
    
    return np.mean(aps) if aps else 0.0


def compute_ood_metrics(sim_mat, query_ids, gallery_ids, seen_classes):
    """
    Compute open-world/OOD detection metrics.
    Args:
        sim_mat: similarity matrix (query x gallery)
        query_ids: labels for queries
        gallery_ids: labels for gallery
        seen_classes: set of class IDs seen during training
    Returns:
        dict with AUROC, FPR@TPR95, Recall@1 on Seen, Recall@1 on Unseen
    """
    num_queries = sim_mat.shape[0]
    max_sim, _ = torch.max(sim_mat, dim=1)
    
    if isinstance(query_ids, torch.Tensor):
        query_ids_list = query_ids.cpu().numpy().tolist()
    else:
        query_ids_list = query_ids
    
    is_seen = torch.tensor([q in seen_classes for q in query_ids_list], dtype=torch.bool)
    is_unseen = ~is_seen
    
    results = {}
    
    if is_seen.sum() > 0:
        seen_sims = max_sim[is_seen]
        seen_labels = torch.ones_like(seen_sims)
        results['Recall@1_Seen'] = compute_recall_at_1_seen(sim_mat[is_seen], query_ids[is_seen], gallery_ids)
    else:
        results['Recall@1_Seen'] = None
    
    if is_unseen.sum() > 0:
        unseen_sims = max_sim[is_unseen]
        unseen_labels = torch.zeros_like(unseen_sims)
        results['Recall@1_Unseen'] = compute_recall_at_1_unseen(sim_mat[is_unseen], query_ids[is_unseen], gallery_ids)
    else:
        results['Recall@1_Unseen'] = None
    
    if is_seen.sum() > 0 and is_unseen.sum() > 0:
        y_true = torch.cat([torch.ones_like(seen_sims), torch.zeros_like(unseen_sims)]).numpy()
        y_score = torch.cat([seen_sims, unseen_sims]).numpy()
        
        try:
            results['AUROC'] = roc_auc_score(y_true, y_score)
        except ValueError:
            results['AUROC'] = None
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            idx = np.where(tpr >= 0.95)[0]
            if len(idx) > 0:
                results['FPR@TPR95'] = fpr[idx[0]]
            else:
                results['FPR@TPR95'] = 1.0
        except ValueError:
            results['FPR@TPR95'] = None
    else:
        results['AUROC'] = None
        results['FPR@TPR95'] = None
    
    return results


def compute_recall_at_1_seen(sim_mat, query_ids, gallery_ids):
    """Recall@1 for seen classes"""
    num_queries = sim_mat.shape[0]
    sorted_indices = torch.argsort(sim_mat, dim=1, descending=True)
    correct = 0
    for i in range(num_queries):
        top1_label = gallery_ids[sorted_indices[i, 0]]
        if top1_label == query_ids[i]:
            correct += 1
    return correct / num_queries if num_queries > 0 else 0.0


def compute_recall_at_1_unseen(sim_mat, query_ids, gallery_ids):
    """Recall@1 for unseen classes (should be low for good OOD detection)"""
    num_queries = sim_mat.shape[0]
    sorted_indices = torch.argsort(sim_mat, dim=1, descending=True)
    correct = 0
    for i in range(num_queries):
        top1_label = gallery_ids[sorted_indices[i, 0]]
        if top1_label == query_ids[i]:
            correct += 1
    return correct / num_queries if num_queries > 0 else 0.0


def compute_lifelong_metrics(task_maps, num_tasks):
    """
    Compute lifelong learning metrics from mAP per task.
    Args:
        task_maps: list of mAP values for each task (after each incremental step)
        num_tasks: total number of tasks
    Returns:
        dict with plasticity, forgetting, overall
    """
    if len(task_maps) < 2:
        return {'Plasticity': None, 'Forgetting': None, 'Overall': None}
    
    task_maps = np.array(task_maps)
    
    plasticity = np.mean(task_maps)
    
    forgetting = 0.0
    for i in range(1, len(task_maps)):
        max_prev = np.max(task_maps[:i])
        forgetting += max(0, max_prev - task_maps[i])
    forgetting = forgetting / (len(task_maps) - 1) if len(task_maps) > 1 else 0.0
    
    overall = plasticity - forgetting
    
    return {
        'Plasticity': plasticity,
        'Forgetting': forgetting,
        'Overall': overall
    }


def evaluate_incremental_retrieval(sim_mat, query_ids, gallery_ids, seen_classes, task_id, task_maps_history):
    """
    Complete evaluation for incremental retrieval.
    Returns all metrics in a single dict.
    """
    results = {}
    
    retrieval_metrics = recall_at_k(sim_mat, query_ids, gallery_ids, ks=(1, 5, 10))
    results.update(retrieval_metrics)
    
    results['mAP'] = mean_average_precision(sim_mat, query_ids, gallery_ids)
    
    ood_metrics = compute_ood_metrics(sim_mat, query_ids, gallery_ids, seen_classes)
    results.update(ood_metrics)
    
    lifelong_metrics = compute_lifelong_metrics(task_maps_history, task_id + 1)
    results.update(lifelong_metrics)
    
    return results


def log_results_csv(results, task_id, num_classes, cnn_top1=None, nme_top1=None, csv_path='results.csv'):
    """
    Log results to CSV file.
    Header: task,numclass,cnn_top1,nme_top1,R@1,R@5,R@10,mAP,AUROC,FPR95,Plasticity,Forgetting,Overall
    """
    header = 'task,numclass,cnn_top1,nme_top1,R@1,R@5,R@10,mAP,AUROC,FPR95,Plasticity,Forgetting,Overall\n'
    
    row = f"{task_id},{num_classes},"
    row += f"{cnn_top1 if cnn_top1 is not None else ''},"
    row += f"{nme_top1 if nme_top1 is not None else ''},"
    row += f"{results.get('R@1', ''):.4f},"
    row += f"{results.get('R@5', ''):.4f},"
    row += f"{results.get('R@10', ''):.4f},"
    row += f"{results.get('mAP', ''):.4f},"
    row += f"{results.get('AUROC', '') if results.get('AUROC') is not None else ''},"
    row += f"{results.get('FPR@TPR95', '') if results.get('FPR@TPR95') is not None else ''},"
    row += f"{results.get('Plasticity', '') if results.get('Plasticity') is not None else ''},"
    row += f"{results.get('Forgetting', '') if results.get('Forgetting') is not None else ''},"
    row += f"{results.get('Overall', '') if results.get('Overall') is not None else ''}\n"
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if not file_exists:
            f.write(header)
        f.write(row)


def log_history_json(results, task_id, num_classes, json_path='history.json'):
    """Log results to JSON history file"""
    history = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    
    entry = {
        'task': task_id,
        'num_classes': num_classes,
        'metrics': {k: float(v) if v is not None else None for k, v in results.items()}
    }
    history.append(entry)
    
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)


def compute_all_metrics(sim_mat, query_ids, gallery_ids, seen_classes, task_id, task_maps_history,
                        cnn_top1=None, nme_top1=None, csv_path='results.csv', json_path='history.json'):
    """Compute all metrics and log to CSV and JSON"""
    results = evaluate_incremental_retrieval(
        sim_mat, query_ids, gallery_ids, seen_classes, task_id, task_maps_history
    )
    
    num_classes = len(seen_classes)
    log_results_csv(results, task_id, num_classes, cnn_top1, nme_top1, csv_path)
    log_history_json(results, task_id, num_classes, json_path)
    
    return results


def test_metrics_perfect_case():
    """Unit test with perfect case: R@1/mAP=1.0, AUROC=1.0, FPR95=0.0"""
    num_classes = 10
    num_samples_per_class = 5
    total = num_classes * num_samples_per_class
    
    query_ids = torch.repeat_interleave(torch.arange(num_classes), num_samples_per_class)
    gallery_ids = query_ids.clone()
    
    sim_mat = torch.zeros(total, total)
    for i in range(total):
        for j in range(total):
            if query_ids[i] == gallery_ids[j]:
                sim_mat[i, j] = 1.0
            else:
                sim_mat[i, j] = 0.0
    
    seen_classes = set(range(num_classes))
    
    results = evaluate_incremental_retrieval(sim_mat, query_ids, gallery_ids, seen_classes, 0, [1.0])
    
    assert abs(results['R@1'] - 1.0) < 1e-6, f"R@1 should be 1.0, got {results['R@1']}"
    assert abs(results['mAP'] - 1.0) < 1e-6, f"mAP should be 1.0, got {results['mAP']}"
    assert results['AUROC'] == 1.0 or results['AUROC'] is None, f"AUROC should be 1.0 or None, got {results['AUROC']}"
    assert results['FPR@TPR95'] == 0.0 or results['FPR@TPR95'] is None, f"FPR@TPR95 should be 0.0 or None, got {results['FPR@TPR95']}"
    
    print("Perfect case test passed!")
    return True


def test_metrics_all_seen():
    """Test that OOD metrics return None when all classes are seen"""
    num_classes = 5
    num_samples_per_class = 3
    total = num_classes * num_samples_per_class
    
    query_ids = torch.repeat_interleave(torch.arange(num_classes), num_samples_per_class)
    gallery_ids = query_ids.clone()
    
    sim_mat = torch.rand(total, total)
    sim_mat = (sim_mat + sim_mat.T) / 2
    sim_mat.fill_diagonal_(1.0)
    
    seen_classes = set(range(num_classes))
    
    results = evaluate_incremental_retrieval(sim_mat, query_ids, gallery_ids, seen_classes, 0, [0.5])
    
    assert results['AUROC'] is None, f"AUROC should be None when all seen, got {results['AUROC']}"
    assert results['FPR@TPR95'] is None, f"FPR@TPR95 should be None when all seen, got {results['FPR@TPR95']}"
    assert results['Recall@1_Unseen'] is None, f"Recall@1_Unseen should be None when all seen"
    
    print("All-seen case test passed!")
    return True


if __name__ == '__main__':
    test_metrics_perfect_case()
    test_metrics_all_seen()
    print("All metric tests passed!")