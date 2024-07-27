import torch
from PIL import Image
import pandas as pd
import os
import tqdm as tqdm
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


TRAIN_PATH = 'data/train/'
TEST_PATH = 'data/train/'
TRAIN_CSV_DATA = 'data/Training_set.csv'
np.random.seed(42)
TRAIN_RATIO = 0.8




def get_transforms():
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor()])

    return {'train' : train_transforms,
                'val' : val_transforms}

class Dataset:
    def __init__(self, train_path, df, transform):
        self.images_path = train_path
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        label = self.df.iloc[idx]['label']
        is_train = self.df.iloc[idx]['is_train']
        img = Image.open(os.path.join(self.images_path, img_name))

        if self.transform:
            if is_train:
                img = self.transform['train'](img)
            else:
                img = self.transform['val'](img)
            
        return img, torch.tensor(label)
    
def prepare_data(transforms):
    
    df = pd.read_csv(TRAIN_CSV_DATA)
    df['is_train'] = np.random.choice([1, 0], 
                                      size=len(df), 
                                      p=[TRAIN_RATIO, 1- TRAIN_RATIO])

    labels = df['label'].unique()
    
    df['label'] = df['label'].replace(labels, range(len(labels)))

    df_train = df[df['is_train'] == 1]
    df_val = df[df['is_train'] == 0]
    
    train_dataset = Dataset(TRAIN_PATH, df_train, transforms)
    val_dataset = Dataset(TRAIN_PATH, df_val, transforms)
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                shuffle=True,
                                                batch_size = 16,
                                                num_workers=0)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                shuffle=False,
                                                batch_size = 16,
                                                num_workers=0)

    return train_dataloader, val_dataloader, len(labels)

def train_eval_epoch(model, dataloader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for inputs, labels in tqdm.tqdm(dataloader, desc='Train' if is_training else 'Eval'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if is_training:
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return model, epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1):
    for epoch in range(num_epochs):
        
        model, train_loss, train_acc = train_eval_epoch(model=model, dataloader=train_loader, 
                                                        criterion=criterion, optimizer=optimizer, 
                                                        device=device, is_training=True)
        model, val_loss, val_acc = train_eval_epoch(model=model, dataloader=val_loader, 
                                                    criterion=criterion, optimizer=None,
                                                    device=device, is_training=False)
        
        
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')

    return model

def main():
    
    transforms = get_transforms()
    
    train_dataloader, val_dataloader, num_classes = prepare_data(transforms) 
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=10)

if __name__ == '__main__':
    main()
