from config.model_parameters import *
from model.crnn import CRNN
from utils.load_data import get_dataloader
from utils.view_results import plot_loss, print_prediction

import torch.nn as nn
import numpy as np





class TrainCRNN:
    def __init__(self , data_dir) :

        self.train_dataloader , self.val_dataloader ,self.train_dataset , self.val_dataset  = get_dataloader(data_dir)  

        self.train_losses = []
        self.val_losses = []
        self.val_epoch_len = len(self.val_dataset) // BATCH_SIZE
        self.val_epoch_len

        self.crnn = CRNN(hidden_size=hidden_size, vocab_size=vocab_size, 
                    bidirectional=bidirectional, dropout=dropout).to(device)
    
        self.lr = 0.02
        self.optimizer = torch.optim.SGD(self.crnn.parameters(), lr=self.lr, nesterov=True, 
                                    weight_decay=weight_decay, momentum=momentum)
        self.critertion = nn.CTCLoss(blank=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, patience=5)


        self.epoch = 0
        self.max_epoch = 50


    def validation(self , model, val_losses, label_converter):
        with torch.no_grad():
            model.eval()
            for batch_img, batch_text in self.val_dataloader:
                logits = self.crnn(batch_img.to(device))
                val_loss = self.calculate_loss(logits, batch_text, label_converter)
                val_losses.append(val_loss.item())
        return val_losses
    

    def calculate_loss(self , logits, texts, label_converter):
        # get infomation from prediction
        device = logits.device
        input_len, batch_size, vocab_size = logits.size()
        # encode inputs
        logits = logits.log_softmax(2)
        encoded_texts, text_lens = label_converter.encode(texts)
        logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        # calculate ctc
        loss = self.critertion(logits, encoded_texts, 
                        logits_lens.to(device), text_lens)
        return loss
    



    def train(self):
         while self.epoch <= self.max_epoch:
            self.crnn.train()
            for idx, (batch_imgs, batch_text) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                logits = self.crnn(batch_imgs.to(device))
                # calculate loss
                train_loss = self.calculate_loss(logits, batch_text, label_converter)
                if np.isnan(train_loss.detach().cpu().numpy()):
                    continue
                self.train_losses.append(train_loss.item())
                # make backward
                train_loss.backward()

                nn.utils.clip_grad_norm_(self.crnn.parameters(), clip_norm)
                self.optimizer.step()

            val_losses = self.validation(self.crnn, val_losses, label_converter)
            
            # printing progress
            plot_loss(epoch, self.train_losses, val_losses)
            print_prediction(self.crnn, self.val_dataset, device, label_converter)
            
            self.scheduler.step(val_losses[-1])
            epoch += 1

         return self.crnn
    


if __name__ == "__main__":
    
    data_dir = "data/OCR_Text_Dataset/OCR_Text/"
    model_train = TrainCRNN(data_dir)

    model_train.train()