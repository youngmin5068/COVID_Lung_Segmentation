import torch
import torch.nn as nn
import torch.nn.functional as F
from dice_loss import dice_coeff


def eval_net(
        net, 
        val_loader, 
        device,
        deep_supervision
            ):
    
    print('validation Start')
    net.eval()
    n_val = len(val_loader)
    tot = 0

    for imgs, true_masks in val_loader:
        imgs = imgs.to(device,dtype=torch.float32)
        true_masks = true_masks.to(device,dtype=torch.float32)

        with torch.no_grad():
            if deep_supervision:
                mask_preds = net(imgs)
                mask_pred = mask_preds[-1]

            else:
                mask_pred = net(imgs)
        
        if net.num_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()

            tot += dice_coeff(pred, true_masks).item()
    
    net.train()
    return tot / n_val


