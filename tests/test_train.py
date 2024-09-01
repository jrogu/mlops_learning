import torch
import pandas as pd
import unittest
import random
import os
from torchvision import transforms
from PIL import Image
import shutil

from train import create_model, prepare_data, get_transforms


TRAIN_IMAGES_PATH = 'data/tmp_train/'
CSV_PATH = 'data/tmp_csv.csv'
NUM_TMP_RECORDS = 3


def generate_random_image(filename, width=100, height=100):
    # Create a random image with RGB colors
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    
    for i in range(width):
        for j in range(height):
            # Assign random colors to each pixel
            pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    image.save(filename)
    
def generate_csv(path):
    
    data = {
        'filename': ['Image_1.jpg', 'Image_2.jpg', 'Image_3.jpg', 'Image_4.jpg', 'Image_5.jpg'],
        'label': ['SOUTHERN DOGFACE', 'ADONIS', 'BROWN SIPROETA' ,'BROWN SIPROETA', 'BROWN SIPROETA']
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)    
    

def set_up_tmp_data():
    os.makedirs(TRAIN_IMAGES_PATH, exist_ok=True)
    generate_csv(CSV_PATH)
    for i in range(1, NUM_TMP_RECORDS + 6):
        generate_random_image(os.path.join(TRAIN_IMAGES_PATH, f'Image_{i}.jpg'))

set_up_tmp_data()

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.num_classes = 75
        self.tmp_img = torch.rand(1, 3, 224, 224)

        self.train_ratio = 0.7
        self.batch_size = 3
    
    def tearDown(self):
        if os.path.exists(TRAIN_IMAGES_PATH):
            shutil.rmtree(TRAIN_IMAGES_PATH)
        if os.path.exists(CSV_PATH):
            os.remove('data/tmp_csv.csv')
        
    def test_model_output_shape(self):
        model = create_model(num_classes=self.num_classes, device='cpu')
        output = model(self.tmp_img)
        
        self.assertEqual(output.shape, torch.Size([1, self.num_classes]))
        
    def test_dataset_and_dataloader(self):
        train_tmp_num_data = int(self.batch_size * self.train_ratio)
        
        transforms = get_transforms()
        train_dataloader, val_dataloader, num_classes = prepare_data(transforms, 
                                                                TRAIN_IMAGES_PATH, 
                                                                CSV_PATH, 
                                                                self.train_ratio,
                                                             self.batch_size)
    
        # Get batchsize from image becouse of randomness and small tmp data size TODO
        for img, label in train_dataloader:
            self.assertEqual(img.shape, torch.Size([img.shape[0], 3, 224, 224]))
            self.assertEqual(label.shape, torch.Size([img.shape[0]]))
            break
        for img, label in val_dataloader:
            self.assertEqual(img.shape, torch.Size([img.shape[0], 3, 224, 224]))
            self.assertEqual(label.shape, torch.Size([img.shape[0]]))
            break
        
if __name__ == '__main__':
    unittest.main()