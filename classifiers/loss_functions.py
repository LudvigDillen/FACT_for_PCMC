import torch
import torch.nn.functional as F


def emd_loss(logits, labels):
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)

    # One-hot encode labels
    one_hot_labels = torch.zeros_like(probs)
    one_hot_labels[torch.arange(labels.size(0)), labels] = 1

    # Compute cumulative sums
    cdf_probs = torch.cumsum(probs, dim=1)
    cdf_one_hot_labels = torch.cumsum(one_hot_labels, dim=1)

    # Compute loss by directly subtracting cumulative sums and taking mean of absolute differences
    loss_per_sample = torch.sum(torch.abs(cdf_probs - cdf_one_hot_labels), dim=1)
    average_loss = torch.mean(loss_per_sample)
    return average_loss


def get_loss(mode, cross_entropy, pred, target, gamma_lf):
    if mode == "emd":
        loss = emd_loss(pred, target)
    elif mode == "emd_cross_entropy":
        loss_cross_entropy = cross_entropy(pred, target)
        loss_emd = emd_loss(pred, target)
        loss = loss_cross_entropy + gamma_lf*loss_emd
    elif mode == "cross_entropy":
        loss = cross_entropy(pred, target)
    else:
        exit("ERROR: Loss function mode invalid")
    return loss
