"""
Loss functions for Exphormer_Max.
"""
import torch
import torch.nn.functional as F


def kgc_full_graph_ce(scores, true_tails, filter_dict, chunk_h, chunk_r,
                      label_smoothing=0.0, head_filter=None, base_num_rel=None):
    """
    Filtered softmax cross-entropy loss for full-graph KGC training.

    For each query (h, r, t):
      - Mask out all other known true answers (set scores to -inf before softmax).
      - Compute: loss = -log_softmax(masked_scores)[t]

    With label_smoothing > 0 the loss is:
      loss = (1 - eps) * NLL_true + eps * mean(-log_probs over all entities)

    Reciprocal queries (r >= base_num_rel) use head_filter[(h, r_orig)] to mask
    all known heads across train+val+test, not just training heads.

    Args:
        scores          (B, N)  float  — raw logits from the model.
        true_tails      (B,)    long   — global entity index of the true tail.
        filter_dict     dict    — (h, r) -> set(all known tails) from KGCDataset.
        chunk_h         list[int] — head entity for each query in the batch.
        chunk_r         list[int] — relation for each query in the batch.
        label_smoothing float   — smoothing coefficient eps in [0, 1).
        head_filter     dict    — (t, r_orig) -> set(all known heads), for reciprocal.
        base_num_rel    int     — number of base relations; r >= base_num_rel = reciprocal.

    Returns:
        (loss scalar, scores (B, N))
    """
    B, N = scores.shape
    device = scores.device

    masked = scores.clone()
    for i, (h, r, t) in enumerate(zip(chunk_h, chunk_r, true_tails.tolist())):
        if base_num_rel is not None and r >= base_num_rel and head_filter is not None:
            # Reciprocal query (t_orig, r_inv, h_orig): filter all known heads.
            r_orig = r - base_num_rel
            for kt in head_filter.get((h, r_orig), set()):
                if kt != t:
                    masked[i, kt] = float('-inf')
        else:
            for kt in filter_dict.get((h, r), set()):
                if kt != t:
                    masked[i, kt] = float('-inf')

    log_probs = F.log_softmax(masked, dim=-1)          # (B, N)
    nll = -log_probs[torch.arange(B, device=device), true_tails]  # (B,)

    if label_smoothing > 0.0:
        # Smoothed CE: (1-eps)*NLL_true + eps*mean(-log_probs)
        # Clamp log_probs to avoid -inf * smoothing_weight = nan for filtered entities.
        # Filtered entities have exp(log_p)≈0 so they barely contribute to the mean.
        smooth_reg = -log_probs.clamp(min=-20.0).mean(dim=-1)   # (B,)
        loss = ((1.0 - label_smoothing) * nll + label_smoothing * smooth_reg).mean()
    else:
        loss = nll.mean()

    return loss, scores


def compute_loss(pred, true, loss_fun):
    """
    Dispatch loss computation based on loss_fun name.

    Returns:
        loss (scalar tensor)
        pred_score (tensor for metric computation)
    """
    if loss_fun == 'cross_entropy':
        if pred.ndim > 1 and true.ndim == 1:
            pred_score = F.log_softmax(pred, dim=-1)
            loss = F.nll_loss(pred_score, true.long())
        elif pred.ndim == 1 or (pred.ndim > 1 and pred.shape[1] == 1):
            pred = pred.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, true.float())
            pred_score = torch.sigmoid(pred)
        else:
            raise ValueError(f"Unexpected pred shape: {pred.shape}")

    elif loss_fun == 'weighted_cross_entropy':
        assert pred.ndim > 1, "weighted_cross_entropy requires 2D pred"
        pred_score = F.log_softmax(pred, dim=-1)
        # Compute class weights from label frequencies
        n_classes = pred.shape[1]
        true_flat = true.view(-1).long()
        counts = torch.bincount(true_flat, minlength=n_classes).float()
        weight = (counts.sum() / (counts + 1e-6)).to(pred.device)
        weight = weight / weight.sum() * n_classes
        loss = F.nll_loss(pred_score, true_flat, weight=weight)

    elif loss_fun == 'l1':
        loss = F.l1_loss(pred.squeeze(-1), true.float())
        pred_score = pred

    elif loss_fun == 'mse':
        loss = F.mse_loss(pred.squeeze(-1), true.float())
        pred_score = pred

    else:
        raise ValueError(f"Unknown loss function: {loss_fun}")

    return loss, pred_score
