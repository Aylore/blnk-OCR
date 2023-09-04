
import os
from PIL import Image
import torch
from torch.utils.data import  Dataset
from torchvision import transforms
import collections.abc as collections

from src.text_preprocessing import TextProcessing


class OCRDataset(Dataset):
    def __init__(self,  data):

        """
            data : dataframe with two columns `image` , `text` 
        """

#         pathes = os.listdir(img_dir)
#         abspath = os.path.abspath(img_dir)
        
#         self.img_dir = img_dir
        self.process_text = TextProcessing()
        pathes = data["image"].values.tolist()
        self.pathes = data["image"].values.tolist()
        
#         self.pathes = [os.path.join(abspath, path) for path in pathes]
        self.list_transforms = transforms.Compose([ transforms.Resize((50, 224)), 
                                                   transforms.ToTensor()])
        
    def __len__(self):
        return len(self.pathes)
    
    def __getitem__(self, idx):
        path = self.pathes[idx]
        text = self.process_text.apply_all(self.read_text(path[:-3] + "txt"))       #self.get_filename(path)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, text
    
    
    def read_text(self , file_path):
        with open(file_path, 'r') as txt_file:
                text = txt_file.read()

        return text

    def get_filename(self, path: str) -> str:
        return os.path.basename(path).split('.')[0].lower().strip()
    
    def transform(self, img) -> torch.Tensor:
        return self.list_transforms(img)