import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import random
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
def dice_score(pred_mask_f, gt_mask_f, epsilon = 1e-6):
    # implement the Dice score here
    with torch.no_grad():
        pred_mask_f = F.sigmoid(pred_mask_f).to(device)
        pred_mask_f = (pred_mask_f > 0.5).float()
        intersection = torch.sum(pred_mask_f * gt_mask_f, dim = 1)
        dice_score = (2. * intersection + epsilon) / (torch.sum(pred_mask_f, dim = 1) + torch.sum(gt_mask_f, dim = 1) + epsilon)
    return dice_score.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, smooth = 1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_mask, gt_mask):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pred_mask = pred_mask.view(pred_mask.size(0), -1)
        gt_mask = gt_mask.view(gt_mask.size(0), -1)
        pred_mask_s = F.sigmoid(pred_mask).to(device)
        intersection = torch.sum(pred_mask_s * gt_mask, dim = 1)
        dice = (2. * intersection + self.smooth) / (torch.sum(pred_mask_s, dim = 1) + torch.sum(gt_mask, dim = 1) + self.smooth)
        # Dice loss
        dice_loss = 1 - dice.mean()
        #try smooth = 1e-6
        
        # Binary Cross-Entropy loss
        print(pred_mask.shape)
        print(gt_mask.shape)
        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask).to(device)

        # Combined loss
       
        return dice_loss + bce_loss

def get_hyperparameters(model):
    assert model in {'unet', 'resnet34_unet'}
    hyperparameters = {}
    if model == 'unet':
        hyperparameters['lr'] = 0.0003
        hyperparameters['n_epochs'] = 50
        hyperparameters['patience'] = 10
        hyperparameters['weight_decay'] = 0.0001
    else:
        hyperparameters['lr'] = 0.001
        hyperparameters['n_epochs'] = 40
        hyperparameters['patience'] = 10
        hyperparameters['weight_decay'] = 0.0001 
    
    return hyperparameters

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot(epochs, plot_list):
    plt.figure(figsize=(14, 6))

    # for loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, plot_list[0], marker='o', linestyle='-', linewidth=2, color='blue', label='Train Loss')
    plt.plot(epochs, plot_list[1], marker='o', linestyle='--', linewidth=2, color='orange', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # for acc
    plt.subplot(1, 2, 2)
    plt.plot(epochs, plot_list[2], marker='o', linestyle='-', linewidth=2, color='blue', label='Train Dice')
    plt.plot(epochs, plot_list[3], marker='o', linestyle='--', linewidth=2, color='orange', label='Validation Dice')
    plt.title('Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
