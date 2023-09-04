from torchvision import transforms
from torch.utils.data import DataLoader 
from config.model_parameters import BATCH_SIZE

from src.dataset import OCRDataset
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader




def get_dataloader(data_dir ):



    files = os.listdir(data_dir)
    samples = []
    for filename in files:
                if filename.endswith('.jpg'):
                    base_name = os.path.splitext(filename)[0]
                    txt_name = base_name + '.txt'
                    if txt_name in files:
                        samples.append( (filename, txt_name) )
    df = pd.DataFrame(samples , columns=["image" , "text"])
    df["image"] = data_dir + df["image"]
    df["text"] = data_dir + df["text"]



    ## Split Data
    df_train , df_val = train_test_split(df , test_size = 0.2 , random_state=3407)

    df_train.reset_index(drop=True , inplace=True)
    df_val.reset_index(drop=True , inplace=True)


    ## Create data set and dataloaders

    train_dataset = OCRDataset(df_train)
    val_dataset = OCRDataset(df_val)


    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                 shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                                 shuffle=False)

    return train_dataloader , val_dataloader , train_dataset ,val_dataset
