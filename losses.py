import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
import pdb

class LogSoftmaxWithProbWrapper(nn.Module):
    """
    Arguments
    ---------
    Returns
    ---------
    loss : torch.Tensor
        Learning loss
    predictions : torch.Tensor
        Log probabilities
    Example
    -------
    """

    def __init__(self, loss_fn):
        super(LogSoftmaxWithProbWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1, outdim].
        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


class LwFDistillationLoss(nn.Module):
    '''
        Distillation loss for LwF (logits)
    '''
    def __init__(self, tau=2):
        super().__init__()
        self.tau = tau
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        outputs_s = F.log_softmax(outputs / self.tau, dim=1)
        targets_s = F.softmax(targets / self.tau, dim=1)
        loss = self.criterion(outputs_s, targets_s) / targets.shape[0]
        return loss

class SimCLRLoss(nn.Module):
    '''
        Loss for SimCLR
    '''
    def __init__(self, tau=0.5):
        super(SimCLRLoss, self).__init__()
        self.tau = tau
        self.criterion = NTXentLoss(temperature=tau)

    def forward(self, z1, z2):
        """
        Arguments
        ---------
        z1 : torch.Tensor (B, D)
            Projected features of augmented examples
        z2 : torch.Tensor (B, D)
            Projected features of the same examples with different augmentations
        Returns
        ---------
        loss : torch.Tensor 
            Scalar NT-Xent loss
        """
        z_pairs = torch.cat([z1, z2], dim=0) # (2B, D)
        indices = torch.arange(0, z1.shape[0], device=z1.device)
        labels = torch.cat([indices, indices], dim=0)

        return self.criterion(z_pairs, labels)

class BarlowTwinsLoss(nn.Module):
    '''
        Loss for Barlow Twins
        Reference: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    '''
    def __init__(self, lambda_rr=5e-3, out_dim=8192, eps=1e-8, loss_scale=1.):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_rr = lambda_rr
        self.out_dim = out_dim
        self.eps = eps
        self.loss_scale = loss_scale

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        """
        Arguments
        ---------
        z1 : torch.Tensor (B, D)
            Projected features of augmented examples
        z2 : torch.Tensor (B, D)
            Projected features of the same examples with different augmentations
        Returns
        ---------
        loss : torch.Tensor 
            Scalar Barlow Twins loss
        """ 
        B, D = z1.shape
        z_1_norm = (z1 - z1.mean(0)) / (z1.std(0) + self.eps)
        z_2_norm = (z2 - z2.mean(0)) / (z2.std(0) + self.eps)
        c = z_1_norm.T @ z_2_norm / B

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_rr * off_diag
        return self.loss_scale * loss
        # return loss / (c.shape[0]*c.shape[1]) * self.loss_scale


class ClapLatentLoss(nn.Module):
    """
        Loss for measuring the divergence between the latents
    """
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_type == 'cosine':
            self.margin = 0
            self.lamb = 1. / 32
        else:
            raise ValueError("Unknown loss type {}".format(loss_type))

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2, sim_fn=None):
        if self.loss_type == 'CrossEntropy':
            sim_mat = sim_fn(z1, z2)
            loss1 = self.criterion(sim_mat, torch.arange(sim_mat.shape[1]).long().to(sim_mat.device))
            loss2 = self.criterion(sim_mat.T, torch.arange(sim_mat.shape[0]).long().to(sim_mat.device))
            loss = 0.5 * (loss1 + loss2)
        elif self.loss_type == 'MSE':
            loss = self.criterion(z1, z2)
        elif self.loss_type == 'cosine':
            z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True) + 1e-8)
            z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True) + 1e-8)
            sim_mat = z1_norm @ z2_norm.T
            on_diag_loss = (1 - torch.diagonal(sim_mat)).sum()
            off_diag_loss = torch.clamp(self.off_diagonal(sim_mat).sum() - self.margin, min=0)
            loss = on_diag_loss + self.lamb * off_diag_loss
        return loss


class MLMLoss(nn.Module):
    """
        adapted from https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/trainer_pt_utils.py#L470
    """
    def __init__(
        self,
        epsilon=0.1,
        ignore_index=-100
    ):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, logits, labels, shift_labels=False):
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class SoftCLAPLoss(nn.Module):
    def __init__(
        self,
        beta,
    ):
        super().__init__()
        self.beta = beta
        self.criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)

    def forward(self, logits, soft_z1, soft_z2):
        if soft_z1 is None or soft_z2 is None:
            target_1 = torch.eye(logits.shape[1]).to(logits.device)
            target_2 = torch.eye(logits.shape[0]).to(logits.device)
        with torch.no_grad():
            soft_target_1 = soft_z1 @ soft_z1.t()
            soft_target_2 = soft_z2 @ soft_z2.t()
            real_target_1 = torch.eye(logits.shape[1]).to(logits.device)
            real_target_2 = torch.eye(logits.shape[0]).to(logits.device)
            target_1 = (1 - self.beta) * real_target_1 + self.beta * soft_target_1
            target_2 = (1 - self.beta) * real_target_2 + self.beta * soft_target_2
        loss_1 = self.criterion(F.log_softmax(logits, dim=-1), target_1)
        loss_2 = self.criterion(F.log_softmax(logits.t(), dim=-1), target_2)
        return 0.5 * (loss_1 + loss_2)
