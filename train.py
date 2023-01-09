import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import albumentations
import albumentations.pytorch
import torch.nn as nn
import time
from network import *
from dataset import *
# from eval import eval_net

SEED = 76075154
dataset_path = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((512,512)),
    ]
)

def train_net(net,
              device,
              epochs=10,
              batch_size=64,
              lr=0.0001,
              val_percent=0.004,
              save_cp=True,
              deep_supervision=False
              ):

    dataset = seg_Dataset(dataset_path,transform=transform)
    n_val = int(len(dataset)*val_percent)
    n_train = len(dataset) - n_val
    train,val = random_split(dataset,[n_train,n_val])
 

    train_loader = DataLoader(train, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val,batch_size=batch_size,shuffle=True)
    global_step = 0

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode= 'min' if net.n_classes > 1 else 'max',patience=3,factor=0.99)
    
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() 
    
    for epoch in range(epochs):
        start = time.time()
        net.train()

        i = 1
        epoch_loss = 0.0
        eval_count = 0

        for (imgs,true_masks) in train_loader:
            try:
                imgs = imgs.to(device=device,dtype=torch.float32)
                true_masks = true_masks.to(device=device,dtype=torch.float32)
                
            except Exception as e:
                print(e)

            if deep_supervision:
                masks_preds = net(imgs)
                loss=0
                for masks_pred in masks_preds:
                    loss += criterion(masks_pred,true_masks)
            else:
                masks_preds = net(imgs)
                loss = criterion(masks_preds,true_masks)
            
            epoch_loss += loss.item()

            if i*batch_size%100 ==0:
                print('epoch : {}, index : {}/{}, loss (batch) : {:.4f}'.format(epoch+1, i*batch_size,n_train, loss.item()))
            i += 1
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1) 
        optimizer.step()
        
        global_step += 1

            # if global_step % (len(dataset) // (10 * batch_size)) == 0:
            #     eval_count += 1
            #     val_score = eval_net(net, val_loader, device,epoch,eval_count,deep_supervision=deep_supervision)
            #     print('Validation Dice Coeff : {:.4f}'.format(val_score))
            #     scheduler.step(val_score)
        dir_checkpoint = '/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/model'

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch+1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
        end = time.time()
        print('Epoch time : ' + str((end - start) // 60))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    #device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    #device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    #torch.cuda.set_device(device)
    logging.info(f'Using device {device}')

    net = UNet(input_channels=1, n_classes=1, bilinear=True)
    #net = SEUNet(input_channels=1, num_classes=1, bilinear=True)
    #net = NestedUNet(num_classes=1,input_channels=1, deep_supervision=False)
    #net = UNet_3Plus(input_channels=1,num_classes=1)
    #net = UNet_3Plus_DeepSup_CGM(input_channels=1,num_classes=1)


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  deep_supervision=False)