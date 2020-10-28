import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = str(os.path.dirname(os.path.realpath(__file__)))
