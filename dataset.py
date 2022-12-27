import pydicom as dcm
import numpy as np
from PIL import Image
import torch as torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import glob
import os



class seg_Dataset(Dataset):
    def __init__(self,path, transform = False):
        self.path = path
        self.transform = transform

        self.train_path_list = []
        self.train_list = []

        self.label_path_list = []
        self.label_list = []


        self.train_path = path + "/train"
        self.label_path = path + "/label"
        
        for file in os.listdir(self.train_path):
            self.train_path_list.append(os.path.join(self.train_path,file))
        self.train_path_list.sort()
            
        for file in os.listdir(self.label_path):
            self.label_path_list.append(os.path.join(self.label_path,file))           
        self.label_path_list.sort()

    def __len__(self):
        return len(self.train_path_list)
        
    def __getitem__(self,idx):
        image_path = self.train_path_list[idx]
        image = np.array(dcm.dcmread(image_path).pixel_array, dtype="f8")
        
        
        label_path = self.label_path_list[idx]
        label = np.array(Image.open(label_path))

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    
if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512,512)),
        ]
    )
    path = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008"

    dataset = seg_Dataset(path=path, transform=transform)
    train_loader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=True)

images, label = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(images.numpy().squeeze(),cmap='gray')
plt.subplot(1,2,2)
plt.imshow(label.numpy().squeeze(),cmap='gray')
plt.show()
