
import torch

from src.text_encoding import strLabelConverter

alphabet = "ةأىآإ ابتثجحخدذرزسشصضطظعغفقكلمنهوي٠١٢٣٤٥٦٧٨٩0123456789ئءؤ" 

hidden_size = 256
vocab_size = len(alphabet) + 1 # extra character for blank symbol
bidirectional = True
dropout = 0.1
weight_decay = 1e-5
momentum = 0.9
clip_norm = 1
max_epoch = 50
BATCH_SIZE = 8



label_converter = strLabelConverter(alphabet)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
