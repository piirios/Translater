import torch

from data import get_data
from utils import translate_sentence
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, checkpointManager, printc
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data import get_data
from models import Transformer
import os
import json
from pkg_resources import resource_stream

CONF_FILE ="conf.json"
FOLDER_OF_PROJECT = os.path.dirname(os.path.abspath(__file__))

def read_conf():
    with resource_stream('translater', 'conf.json') as f:
        return json.load(f)

def write_conf(data):
    with resource_stream('translater', 'conf.json') as f:
        json.dump(data, f)

def save_model_params(filename, **params):
    data = {i:j for i,j in params.items()}
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_model_params(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data

def check_if_weight_exists(conf, weight_file_path, conf_file_path):
        if not os.path.exists(weight_file_path) or not os.path.exists(conf_file_path):
            return False
        else:
            return True


class Translater:
    def __init__(self, from_lang, to_lang):
        self.fl = from_lang
        self.tl = to_lang
        self.conf = read_conf()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        _, _, self.source, self.target, self.spacy_src, _ = get_data("fr", "en", self.conf['dataset_folder'], "train")

        weight_file_path = os.path.join(self.conf['weight_folder'], f"{from_lang}to{to_lang}2.pth")
        conf_file_path = os.path.join(self.conf['weight_folder'], f"{from_lang}to{to_lang}.conf")

        if check_if_weight_exists(self.conf, weight_file_path, conf_file_path):
            self.trained = True
            self.model = Transformer(*load_model_params(conf_file_path).values(), self.device)
            state = torch.load(weight_file_path)
            self.model.load_state_dict(state['state_dict'])
        else:
            raise ValueError(f"""
            weight didn't for {self.fl}2{self.tl}
            Please do:
                Translater train -fl {self.fl} -tl {self.tl}
        """)

    def __call__(self, sentence):
        translated = translate_sentence(self.model, sentence, self.source, self.target, self.spacy_src, self.device)
        result = ' '.join(translated)
        return result

if __name__ == "__main__":
    tran1 = Translater("fr", "en")
    r = tran1("Bonjour a tous")
    print(r)
    tran2 = Translater("en", "ne")