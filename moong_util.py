import random
import os
import numpy as np
import torch

BASE_DIR = '/content/drive/MyDrive/NLP/ENG/Jigsaw2/'
WV_DIR = '/content/drive/MyDrive/NLP/ENG/wordvector/'
MODEL_DIR = BASE_DIR + 'model/'
DATA_DIR = BASE_DIR + 'input/'
OUTPUT_DIR = BASE_DIR + 'output/'


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
