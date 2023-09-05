from config.model_parameters import *
from model.crnn import CRNN
from src.text_encoding import strLabelConverter

from torchvision import transforms
from utils.decode_preds import decode_prediction
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

def process_image(image):
    transform = transforms.Compose([ transforms.Resize((50, 200)), 
                                                   transforms.ToTensor()])
    img = image.convert('RGB')

    img = transform(img)

    return img





def print_prediction(model, img, device, label_converter):
    
    
    with torch.no_grad():
        model.eval()
        
        img = img.unsqueeze(0)
        logits = model(img.to(device))
        
    pred_text = decode_prediction(logits.cpu(), label_converter)

    img = img.squeeze(0)

    return pred_text
    # title = f' Pred: {pred_text}'
    # plt.imshow(img)
    # plt.title(title)
    # plt.axis('off');

def load_model():
    model = CRNN(hidden_size=hidden_size, vocab_size=vocab_size, 
            bidirectional=bidirectional, dropout=dropout).to(device)

    ## Load weights
    weights_path = "model/crnn.pt"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_state_dict(torch.load(weights_path , map_location=torch.device('cpu')) )

    return model



def infer(img):


    model =load_model()


    image = process_image(img)

    
    ## make predictions

    label_converter  = strLabelConverter(alphabet)
    results = print_prediction(model , image , device , label_converter )
    print(results)

    return results