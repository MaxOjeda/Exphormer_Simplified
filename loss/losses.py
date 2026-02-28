"""
Loss functions for Exphormer_Max.
"""
import torch
import torch.nn.functional as F


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
