import torch

from registration.registration_utils import get_error_class
softplus = torch.nn.Softplus()


def classify_pairs(model, points, regression, reg_method):
    classifier = model.eval()
    pred = classifier(
        points, inference=True
    )  # [B, n_classes], here we have a score for each class
    if regression:
        n_samples = pred.shape[0]
        pred_choice = torch.empty(n_samples, device=pred.device, dtype=torch.long)
        pred_Rplus = softplus(pred)
        for i in range(n_samples):
            pred_choice[i] = get_error_class(pred_Rplus[i], reg_method=reg_method)
    else:
        pred_choice = pred.data.max(1)[1]
    return pred_choice, pred  # highest score wins
