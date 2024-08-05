import torch
from torchvision import transforms
import pandas as pd
from train import create_model, prepare_data, get_transforms
import unittest

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        self.num_classes = 75
        self.tmp_img = torch.rand(1, 3, 224, 224)

        self.train_ratio = 0.8
        self.batch_size = 2
        self.csv_path = 'data/Training_set.csv'
        self.train_images_path = 'data/train/'
        self.batch_size = 16
        
    def test_model_output_shape(self):
        model = create_model(num_classes=self.num_classes, device='cpu')
        output = model(self.tmp_img)
        
        self.assertEqual(output.shape, torch.Size([1, 75]))
        
    def test_dataset_and_dataloader(self):
        transforms = get_transforms()
        # TODO fix paths with tmp images/csv file
        # train_dataloader, val_dataloader, num_classes = prepare_data(transforms, 
        #                                                         self.train_images_path, 
        #                                                         self.csv_path, 
        #                                                         self.train_ratio,
        #                                                         self.batch_size)
        
        # for img, label in train_dataloader:
        #     self.assertEqual(img.shape, torch.Size([self.batch_size, 3, 224, 224]))
        #     self.assertEqual(label.shape, torch.Size([self.batch_size]))
            
        #     break
        
if __name__ == '__main__':
    unittest.main()