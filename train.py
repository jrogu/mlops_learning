import torch
from PIL import Image
import pandas as pd
import os
import tqdm as tqdm
from torchvision.transforms import v2
import numpy as np
from torchvision import transforms

TRAIN_PATH = 'data/train/'
TEST_PATH = 'data/train/'
TRAIN_CSV_DATA = 'data/Training_set.csv'
np.random.seed(42)
TRAIN_RATIO = 0.8

train_transforms = transforms.Compose([transforms.ToTensor()])
val_transforms = transforms.Compose([transforms.ToTensor()])

transforms = {'train' : train_transforms,
              'val' : val_transforms}


class Dataset:
    def __init__(self, train_path, train_csv_path, transform, SPLIT_SIZE):
        self.images_path = train_path
        self.csv_path = train_csv_path
        self.transform = transform
        self.csv = pd.read_csv(train_csv_path)
        self.csv['is_train'] = np.random.choice(
            [1, 0], size=len(self.csv), p=[SPLIT_SIZE, 1- SPLIT_SIZE])
        # categorical to int labels
        self.csv['label'] = self.csv['label'].replace(
            self.csv['label'].unique(), range(len(self.csv['label'].unique()))
            )
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        img_name = self.csv.iloc[idx, 0]
        label = self.csv.iloc[idx, 1]
        is_train = self.csv.iloc[idx, 2]
        img = Image.open(os.path.join(self.images_path, img_name))

        if self.transform:
            if is_train:
                img = self.transform['train'](img)
            else:
                img = self.transform['val'](img)
            
        return img, torch.tensor(label)