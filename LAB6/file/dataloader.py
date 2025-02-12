import torch
import json
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from glob import glob
from torchvision.datasets.folder import default_loader as loader

class iclevr_dataset(Dataset):
    def __init__(self, path,  mode='train'):
        super(iclevr_dataset).__init__()
        assert mode in ['train', 'test', 'new_test']
        self.mode = mode
        self.path = path
        
        with open('file/objects.json', 'r') as file:
            self.objects = json.load(file)
            self.class_num = len(self.objects)
            
        with open(f'file/{mode}.json', 'r') as file:
            self.labels = json.load(file)

        
        if self.mode == 'train':
            self.imgs = glob(os.path.join(self.path,'*.png'))
            self.imgs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]) + \
                10 * int(x.split('/')[-1].split('.')[0].split('_')[-2]))
            self.imgs = self.imgs[:-3]
            #all png file is for training
            
            self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            #resize to 64 * 64 and do normalize in spec
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            img = self.transforms(loader(self.imgs[idx]))
            label_name = self.labels[self.imgs[idx].split('/')[-1]]
            # find the label for the image
            label = [0 for _ in range(self.class_num)]
            for i in label_name: label[self.objects[i]] = 1
            label = torch.tensor(np.array(label), dtype=torch.int32)
            return img, label
        else:
            label = [0 for _ in range(self.class_num)]
            for i in self.labels[idx]: label[self.objects[i]] = 1
            label = torch.tensor(np.array(label), dtype=torch.int32)
            return label
