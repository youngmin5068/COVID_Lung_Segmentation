import logging
import os
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

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Model_SEED = 76075154

dataset_path = "/workspace/test_visual2"
eval_path = "/workspace/eval"

def train_net(net,
              device,
              epochs=10,
              batch_size=4,
              lr=0.0001,
              save_cp=True,   
              deep_supervision=False
              ):

    dataset = seg_Dataset(dataset_path)
    eval_dataset = seg_Dataset(eval_path)
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(eval_dataset,batch_size=batch_size,shuffle=True)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Training size:   {n_train}
        Learning rate:   {lr}        
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')    
    optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=1e-5) # weight_decay : prevent overfitting
    
    if net.num_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss() 

    for epoch in range(epochs):
        eval_count = 0
        start = time.time()
        net.train()
        i=1
    
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

            if i*batch_size%50 == 0:
                print('epoch : {}, index : {}/{}, loss (batch) : {:.4f}'.format(epoch+1, i*batch_size,n_train,loss.item())) 

            i += 1

            if i*batch_size%500 == 0:
                eval_count += 1
                val_score = eval_net(net, val_loader, device,deep_supervision=False)
                print('Validation Dice Coeff : {:.4f}'.format(val_score))


        dir_checkpoint = '/workspace/dir_checkpoint_0302'

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f'/CP_epoch_0302_{epoch+1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
        end = time.time()
        print('Epoch time : ' + str((end - start) // 60))


if __name__ == '__main__':
    torch.manual_seed(Model_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    #net = UNet(input_channels=1,num_classes=1,bilinear=True)
    
    #net = SEUNet(input_channels=1, num_classes=1, bilinear=True)
    net = NestedUNet(num_classes=1,input_channels=1, deep_supervision=False)
    #net = UNet_3Plus(input_channels=1,num_classes=1)
    #net = UNet_3Plus_DeepSup_CGM(input_channels=1,num_classes=1)


    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)

    net.to(device)

    train_net(net=net,device=device)
