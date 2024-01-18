import torch
import torch.nn as nn
import torch.nn.functional as F



class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def consistency_loss(logits_s, logits_w, class_acc,  max_probss, targets_u, name='ce',
                     T=1.0, p_cutoff=0.95, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        # p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
        # pseudo_label = pseudo_label * p_target / p_model
        # pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))
        max_probs, max_idx = max_probss, targets_u
        # print(class_acc[max_idx])
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()
        # print(mask)
        mask_dis = max_probs.lt(p_cutoff).float()



        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask

            # masked_reverse_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask_dis
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss