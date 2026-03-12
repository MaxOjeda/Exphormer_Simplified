"""
Metrics logger for Exphormer_Max.
Replaces graphgym's Logger base class with a standalone implementation.
"""
import json
import logging
import os
import time

import numpy as np
import torch
from scipy.stats import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              mean_absolute_error, mean_squared_error,
                              precision_score, r2_score, recall_score,
                              roc_auc_score)
from torchmetrics.functional import auroc


def accuracy_SBM(targets, pred_int):
    """Accuracy for PATTERN/CLUSTER (benchmarking GNNs)."""
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets_np = targets.cpu().detach().numpy()
    nb_non_empty = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets_np == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty += 1
        else:
            pr_classes[r] = 0.0
    return np.sum(pr_classes) / float(nb_classes)


def eval_spearmanr(y_true, y_pred):
    if y_true.ndim == 1:
        res = [stats.spearmanr(y_true, y_pred)[0]]
    else:
        res = []
        for i in range(y_true.shape[1]):
            mask = ~np.isnan(y_true[:, i])
            res.append(stats.spearmanr(y_true[mask, i], y_pred[mask, i])[0])
    return {'spearmanr': sum(res) / len(res)}


class CustomLogger:
    """Tracks predictions, labels, loss, and computes epoch metrics."""

    def __init__(self, name, task_type, out_dir=None, cfg=None):
        self.name = name
        self.task_type = task_type
        self.out_dir = out_dir
        self.cfg = cfg
        self._reset()

    def _reset(self):
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._lr = 0.0
        self._params = 0
        self._time_used = 0.0
        self._time_total = 0.0
        self._size_current = 0
        self._iter = 0
        self._custom_stats = {}
        self._kgc_ranks = []   # filtered ranks accumulated during KGC eval

    def reset(self):
        self._reset()

    def update_stats(self, true, pred, loss, lr, time_used, params,
                     dataset_name=None, **kwargs):
        batch_size = true.shape[0]
        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = val * batch_size
            else:
                self._custom_stats[key] += val * batch_size

    def update_stats_kgc(self, ranks, time_used, lr=0.0, params=0):
        """
        Accumulate per-query filtered ranks for KGC full-graph evaluation.

        Called once per batch of queries during eval_epoch_kgc.

        Args:
            ranks (list[int] | Tensor): filtered rank of the true answer for
                each query in the batch (1-indexed: rank=1 means top prediction).
            time_used (float): wall time for this batch.
            lr (float): current learning rate.
            params (int): number of model parameters.
        """
        if isinstance(ranks, torch.Tensor):
            ranks = ranks.tolist()
        n = len(ranks)
        self._kgc_ranks.extend(ranks)
        self._size_current += n
        self._iter += 1
        self._time_used += time_used
        self._lr = lr
        self._params = params

    def _get_pred_int(self, pred_score):
        if pred_score.ndim == 1 or pred_score.shape[1] == 1:
            return (pred_score > 0.5).long().squeeze()
        return pred_score.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Task-specific metrics
    # ------------------------------------------------------------------

    def classification_binary(self):
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        reformat = lambda x: round(float(x), max(8, self.cfg.round if self.cfg else 5))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        auroc_score = auroc(pred_score.to(device), true.to(device).long(), task='binary')

        return {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int, zero_division=0)),
            'auc': reformat(auroc_score),
        }

    def classification_multi(self):
        true = torch.cat(self._true)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        reformat = lambda x: round(float(x), max(8, self.cfg.round if self.cfg else 5))
        cfg = self.cfg

        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int, average='macro', zero_division=0)),
        }
        if cfg and cfg.metric_best == 'accuracy-SBM':
            res['accuracy-SBM'] = reformat(accuracy_SBM(true, pred_int))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_classes = pred_score.shape[1]
        if true.shape[0] < 1e7:
            res['auc'] = reformat(auroc(pred_score.to(device),
                                        true.to(device).squeeze().long(),
                                        task='multiclass',
                                        num_classes=n_classes,
                                        average='macro'))
        return res

    def classification_multilabel(self):
        true = torch.cat(self._true)
        pred_score = torch.cat(self._pred)
        reformat = lambda x: round(float(x), max(8, self.cfg.round if self.cfg else 5))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        true_d = true.to(device)
        pred_d = pred_score.to(device)
        n_labels = pred_d.shape[1]
        auroc_score = auroc(pred_d, true_d.long(), task='multilabel', num_labels=n_labels,
                            average='macro')
        pred_bin = (pred_d > 0.0).float()
        acc = (pred_bin == true_d).float().mean()
        return {
            'accuracy': reformat(acc),
            'auc': reformat(auroc_score),
        }

    def regression(self):
        true = torch.cat(self._true).numpy()
        pred = torch.cat(self._pred).numpy()
        reformat = lambda x: round(float(x), max(8, self.cfg.round if self.cfg else 5))
        return {
            'mae': reformat(mean_absolute_error(true, pred)),
            'r2': reformat(r2_score(true, pred, multioutput='uniform_average')),
            'spearmanr': reformat(eval_spearmanr(true, pred)['spearmanr']),
            'mse': reformat(mean_squared_error(true, pred)),
            'rmse': reformat(mean_squared_error(true, pred, squared=False)),
        }

    def kgc_ranking(self):
        """
        Filtered MRR and Hits@K from accumulated per-query ranks.
        Called during full-graph KGC evaluation (val / test).
        """
        reformat = lambda x: round(float(x), max(8, self.cfg.round if self.cfg else 5))
        ranks = torch.tensor(self._kgc_ranks, dtype=torch.float)
        if len(ranks) == 0:
            return {'mrr': 0.0, 'hits@1': 0.0, 'hits@3': 0.0, 'hits@10': 0.0}
        return {
            'mrr':     reformat((1.0 / ranks).mean().item()),
            'hits@1':  reformat((ranks <= 1).float().mean().item()),
            'hits@3':  reformat((ranks <= 3).float().mean().item()),
            'hits@10': reformat((ranks <= 10).float().mean().item()),
        }

    def kgc_train_approx(self):
        """
        Training-time approximation: Hits@1 within the subgraph.
        Uses the padded log-softmax scores accumulated via update_stats().
        Not comparable to filtered eval metrics — used only to track
        whether the model is learning to rank the answer first locally.

        Note: different batches may have different max_N (max nodes in batch),
        so we compute argmax per-batch before concatenating the 1-D hit flags.
        """
        reformat = lambda x: round(float(x), max(8, self.cfg.round if self.cfg else 5))
        hit_flags = []
        for pred_batch, true_batch in zip(self._pred, self._true):
            # pred_batch: (B, max_N_i), true_batch: (B,)
            pred_int = pred_batch.argmax(dim=-1)   # (B,)
            hit_flags.append((pred_int == true_batch).float())
        if not hit_flags:
            return {'subgraph_hits@1': 0.0}
        hit1 = torch.cat(hit_flags).mean().item()
        return {'subgraph_hits@1': reformat(hit1)}

    # ------------------------------------------------------------------
    # Epoch summary
    # ------------------------------------------------------------------

    def write_epoch(self, cur_epoch):
        cfg = self.cfg
        round_n = cfg.round if cfg else 5

        basic = {
            'loss': round(self._loss / max(self._size_current, 1), max(8, round_n)),
            'lr': round(self._lr, max(8, round_n)),
            'params': self._params,
            'time_iter': round(self._time_used / max(self._iter, 1), round_n),
        }

        if self.task_type == 'kgc_ranking':
            # Eval path: full-graph ranks accumulated via update_stats_kgc()
            # Train path: subgraph-level scores accumulated via update_stats()
            if self._kgc_ranks:
                task_stats = self.kgc_ranking()
            else:
                task_stats = self.kgc_train_approx()
        elif self.task_type == 'regression':
            task_stats = self.regression()
        elif self.task_type == 'classification_binary':
            task_stats = self.classification_binary()
        elif self.task_type == 'classification_multi':
            task_stats = self.classification_multi()
        elif self.task_type == 'classification_multilabel':
            task_stats = self.classification_multilabel()
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        stats = {
            'epoch': cur_epoch,
            'time_epoch': round(self._time_used, round_n),
            **basic,
            **task_stats,
        }

        logging.info(f'{self.name}: {stats}')

        if self.out_dir:
            out_path = os.path.join(self.out_dir, f'{self.name}_stats.json')
            _append_json(stats, out_path)

        self.reset()
        return stats

    def close(self):
        pass


def infer_task_type(cfg):
    """Infer task type string from cfg."""
    tt = cfg.dataset.task_type
    if tt == 'kgc_ranking':
        return 'kgc_ranking'
    elif tt == 'classification':
        return 'classification_multi'
    elif tt == 'classification_binary':
        return 'classification_binary'
    elif tt == 'multilabel classification':
        return 'classification_multilabel'
    elif tt == 'regression':
        return 'regression'
    else:
        return 'classification_multi'


def create_loggers(cfg, out_dir=None):
    """Create list of 3 loggers: [train, val, test]."""
    task_type = infer_task_type(cfg)
    loggers = []
    for name in ['train', 'val', 'test']:
        loggers.append(CustomLogger(name=name, task_type=task_type,
                                    out_dir=out_dir, cfg=cfg))
    return loggers


def _append_json(stats, path):
    """Append a stats dict as a line to a JSON-lines file."""
    with open(path, 'a') as f:
        f.write(json.dumps(stats) + '\n')
