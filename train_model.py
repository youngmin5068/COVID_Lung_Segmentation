import argparse
import logging
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from eval import eval_net
from custom_transforms import *
from network import *
from dataset import *
from dice_loss import *
from torch.utils.data import random_split
import gc

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

Model_SEED = 76075154

dataset_path = "/workspace/Covid_Image"


def train_net(net,
              device,
              epochs=10,
              batch_size=4,
              lr=0.0001,
              val_percent = 0.004,
              save_cp=True,   
              deep_supervision=False
              ):

    dataset = seg_Dataset(dataset_path)
    n_train = len(dataset)


    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Training size:   {n_train}
        Validation size: {n_val}
        Learning rate:   {lr}        
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')    
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=1e-5) # weight_decay : prevent overfitting

    if net.num_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() 

    best_acc = 0.0

    for epoch in range(epochs):
        start = time.time()
        net.train()
        i=1
        best_epoch = 1
        
        for imgs,true_masks in train_loader:

            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            if deep_supervision:
                masks_preds = net(imgs)
                loss=0
                for masks_pred in masks_preds:
                    loss += criterion(masks_pred,true_masks)
                loss /= len(masks_preds)
            else:
                masks_preds = net(imgs)
                loss = criterion(masks_preds,true_masks)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)                
            optimizer.step()                

            if i*batch_size%100 == 0:
                print('epoch : {}, index : {}/{}, loss (batch) : {:.4f}'.format(epoch+1, i*batch_size,n_train,loss.item())) 

            i += 1

        #when train epoch end
        print("--------------Validation start----------------")
        net.eval()      
        dice = 0.0
        for imgs, true_masks in val_loader:
            imgs = imgs.to(device=device,dtype=torch.float32)
            true_masks = true_masks.to(device=device,dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred = torch.sigmoid(mask_pred)

            thresh = np.zeros_like(mask_pred.cpu())
            thresh[mask_pred.cpu() > 0.5] = 1

            dice += dice_coefficient(torch.tensor(thresh).cuda(),true_masks)
        print("dice / len(val): {:.4f}/{:.4f}, val_score : {:.4f}".format(dice, len(val_loader),dice/len(val_loader)) )
        
        if dice/len(val_loader) > best_acc:
            best_acc = dice/len(val_loader)
            best_epoch = epoch
        
        dir_checkpoint = '/workspace/dir_checkpoint_Unet++'

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f'/CP_epoch_{epoch+1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
        end = time.time()
        print('Epoch time : ' + str((end - start) // 60))

        print("=====Best Epoch : {}, Best Accuracy : {:.4f}=====".format(best_epoch,best_acc))


if __name__ == '__main__':
    torch.manual_seed(Model_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    net = UNet(input_channels=1,num_classes=1,bilinear=True)
    #net = NestedUNet(num_classes=1,input_channels=1)
    #net = UNet3Plus(n_channels=1,num_classes=1)


    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net, device_ids = [0,1]) 

    net.to(device=device)


    train_net(net=net,device=device,deep_supervision=False)
