"""
This module contains functions for transcribing audio files and streaming audio from the default input device.

The module provides three functions: Index, transcribe, and transcribe_audio. 
1- Index renders the index.html template, which allows users to upload an audio file for transcription. 
2- transcribe transcribes an uploaded audio file and displays the transcript on the index.html template. 
3- transcribe_audio transcribes audio from the default input device and displays the live transcript on the index.html template.

Helper functions for saving and deleting audio files are also included in the helper module.
"""

from unittest import result
from django.shortcuts import render
from django.conf import settings
# from final_pipeline import main
# from .helper import save_audio_file, delete_audio_file
import io, os
import time


from src.inference import infer
from model.crnn import CRNN
from config.model_parameters import *

import PIL

import matplotlib.pyplot as plt

def Index(request):

    """
    Renders the index.html template.

    This function renders the index.html template, which displays a form to upload an audio file and transcribe it.

    Args:
        request: The HTTP request object.

    Returns:
        The HTTP response object containing the rendered index.html template.
    """
    return render(request, "index.html")




def predict(request):
    if request.method == 'POST' and  request.FILES.get("image"):
        ## Load model

        model = CRNN(hidden_size=hidden_size, vocab_size=vocab_size, 
                    bidirectional=bidirectional, dropout=dropout).to(device)
        
        ## Load weights
        weights_path = "model/crnn.pt"
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path , map_location=torch.device('cpu')) )

        

        image = request.FILES['image']

        print("this IS IMAGE", type(image))

        print("THIS IS IMAGE AGAIN " , image)
        
        
        image = PIL.Image.fromarray(plt.imread(image))

        results = infer(image)

        

        return render(request, "index.html", {"prediction": results})
    return render(request , "index.html")


