"""
Loss functions here
new_coteaching: combine small_loss, large_loss and KL_dist together
kl_loss_comput: KL divergence
soft_predict: predict on large_loss set with data augmentation
"""

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# new co-teaching
def new_coteaching(data, targets, model_1, model_2, forgetRate, epoch, init_epoch=5, alpha=0.6):
    Y1 = model_1(data)
    Y2 = model_2(data)
    
    loss_1 = F.nll_loss(Y1, targets, reduce=False).cpu()
    sort_idx_1 = np.argsort(loss_1.data).cuda() # reverse
    loss_1 = loss_1.cuda()
    loss_1_sorted = loss_1[sort_idx_1]

    loss_2 = F.nll_loss(Y2, targets, reduce=False).cpu()
    sort_idx_2 = np.argsort(loss_2.data).cuda() # reverse
    loss_2 = loss_2.cuda()
    loss_2_sorted = loss_2[sort_idx_2]

    rememberRate = 1 - forgetRate
    rememberNum = int(rememberRate * len(loss_1_sorted))

    idx_small_1 = sort_idx_1[:rememberNum]
    idx_small_2 = sort_idx_2[:rememberNum]
    idx_large_1 = sort_idx_1[rememberNum:]
    idx_large_2 = sort_idx_2[rememberNum:]
    if forgetRate > 0.5:
        number = min(epoch/200, 1) * len(idx_large_1)
        idx_large_1 = idx_large_1[:int(number)]
        idx_large_2 = idx_large_2[:int(number)]
    # exchange
    small_loss_1 = F.nll_loss(Y1[idx_small_2], targets[idx_small_2], size_average=False)
    small_loss_2 = F.nll_loss(Y2[idx_small_1], targets[idx_small_1], size_average=False)
    
    if epoch > init_epoch:
#     if epoch >= 0:
        pred_1, kl_1 = soft_predict(data[idx_large_2], model_2)
        large_loss_1 = F.nll_loss(Y1[idx_large_2], pred_1, size_average=False)

        pred_2, kl_2 = soft_predict(data[idx_large_1], model_1)
        large_loss_2 = F.nll_loss(Y2[idx_large_1], pred_2, size_average=False)
    else:
        large_loss_1 = 0
        large_loss_2 = 0
        kl_1 = 0
        kl_2 = 0
    
#     alpha = 1 - forgetRate
#     alpha = 0.5 + epoch/100
    update_1 = small_loss_1 + alpha * large_loss_1 + kl_1
    update_2 = small_loss_2 + alpha * large_loss_2 + kl_2
    
    return update_1, update_2


def kl_loss_compute(pred, soft_targets, reduce=True):
#     kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)
    kl = F.kl_div(pred,soft_targets,reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
    

def soft_predict(im, model, co_lambda=0.5):
    out_1 = model(im)
  
    aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=1),
        torchvision.transforms.RandomRotation(25),
    ])
    x = aug(im)
    out_2 = model(x)
    
    out = out_1 + out_2
    kl_loss = kl_loss_compute(torch.exp(out_1), torch.exp(out_2),reduce=False) + kl_loss_compute(torch.exp(out_2), torch.exp(out_1), reduce=False)
    
    kl_loss = torch.mean(kl_loss)
    
    hard_label = out.data.max(1, keepdim=False)[1]
    
    return hard_label, kl_loss
