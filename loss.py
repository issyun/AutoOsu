import torch

def binary_focal_loss(y, pred, gamma, pos_weight):
    """
    Biary focal loss for when y=1 is the minority class.\n
    INPUT
        gamma: factor for suppressing loss for easy examples (gamma > 1)
        pos_weight: how much to suppress loss when y=0 (0 <= pos_weight <= 1)
    """
    return -(y * (1-pred).pow(gamma) * pred.log() +
                pos_weight * (1-y) * pred.pow(gamma) * (1-pred).log()).mean()

def multi_focal_loss(y, pred, gamma, pos_weight):
    """
    Multi-class focal loss for when y=0 is the majority class.\n
    INPUT
        gamma: factor for suppressing loss for easy examples (gamma > 1)
        pos_weight: how much to suppress loss when y=0 (0 <= pos_weight <= 1)
    """
    p_y = pred[torch.arange(len(pred)), y]
    weight_mask = torch.where(y == 0, pos_weight, 1)
    return -(weight_mask * (1 - p_y).pow(gamma) * p_y.log()).mean()