import click
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
from pathlib import Path


CONF_FILE ="conf.json"
FOLDER_OF_PROJECT = os.path.dirname(os.path.abspath(__file__))

def read_conf():
    with open(os.path.join(FOLDER_OF_PROJECT, CONF_FILE), 'r') as f:
        j = json.load(f)
        j['weight_folder'] = os.sep.join(j['weight_folder'].split(j['sep']))
        j['dataset_folder'] = os.sep.join(j['dataset_folder'].split(j['sep']))
        return j

def write_conf(data):
    data['sep'] = os.sep
    with open(os.path.join(FOLDER_OF_PROJECT, CONF_FILE), 'w') as f:
        json.dump(data, f)
        
@click.group()
def translater():
    pass


def save_model_params(filename, **params):
    data = {i:j for i,j in params.items()}
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_model_params(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data

conf = read_conf()

@translater.command()
@click.argument('folder', type=click.Path(exists=True))
def set_dataset_folder(folder):
    conf['dataset_folder'] = folder
    write_conf(conf)

@translater.command()
@click.argument('folder', type=click.Path(exists=True))
def set_weight_folder(folder):
    conf['weight_folder'] = folder
    write_conf(conf)

@translater.command()
@click.option('--from-lang', '-fl', required=True, type=str)
@click.option('--to-lang', '-tl', required=True, type=str)
@click.option('--sequence', '-sq', required=True, type=str)
def translate(from_lang, to_lang, sequence):
    conf = read_conf()
    _, _, source, target, spacy_src, _ = get_data("fr", "en", conf['dataset_folder'], "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = read_conf()
    weight_file_path = os.path.join(Path(conf['weight_folder']), f"{from_lang}to{to_lang}2.pth")
    conf_file_path = os.path.join(conf['weight_folder'], f"{from_lang}to{to_lang}.conf")

    model = Transformer(*load_model_params(conf_file_path).values(), device)
    state = torch.load(weight_file_path)
    model.load_state_dict(state['state_dict'])

    translated = translate_sentence(model, sequence, source, target,spacy_src, device)

    result = ' '.join(translated)
    click.echo(result)


"""
dataset_folder
weight_folder

num_epochs = 20
learning_rate = 3e-4
batch_size = 32
embedding_size = 1024
num_heads = 6
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.1
max_len = 100
forward_expansion = 2048
"""
def input_of_retrain(fl, tl):
    result = input(f" weight exist of the {fl}2{tl} model, do you want to retrain it? [Y/N]")
    if result == "Y" or result == "y":
        return True
    elif result=="N" or result=="n":
        return False
    else:
        print("please respond with Y or N")
        input_of_retrain(fl, tl)


@translater.command()
@click.option('--from-lang', '-fl', required=True, type=str)
@click.option('--to-lang', '-tl', required=True, type=str)
@click.option('--num-epochs', default=10000, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--learning-rate',  default=3e-4, type=float)
@click.option('--embedding-size',  default=512, type=int)
@click.option('--num-heads',  default=8, type=int)
@click.option('--num-encoder-layers',  default=3, type=int)
@click.option('--num-decoder-layers',  default=3, type=int)
@click.option('--dropout',  default=0.1, type=float)
@click.option('--max-len',  default=200, type=int)
@click.option('--forward-expansion',  default=200, type=int)
def train(from_lang, to_lang, num_epochs, learning_rate, batch_size, embedding_size, num_heads, num_encoder_layers, num_decoder_layers, dropout, max_len, forward_expansion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_file_path = os.path.join(conf['weight_folder'], f"{from_lang}to{to_lang}.conf")

    train_data, valid_data, source, target, spacy_src, spacy_trg = get_data("fr", "en", conf['dataset_folder'], "train")
    if os.path.exists(os.path.join(conf['weight_folder'], f"{from_lang}to{to_lang}.pth")):
        r = input_of_retrain(from_lang, to_lang)
        if r:
            os.remove(os.path.join(conf['weight_folder'], f"{from_lang}to{to_lang}.pth"))
    else:
        r = True
    manager = checkpointManager(os.path.join(conf['weight_folder'], f"{from_lang}to{to_lang}.pth"))


    src_vocab_size = len(source.vocab)
    trg_vocab_size = len(target.vocab)
    src_pad_idx = source.vocab.stoi["<PAD>"]

    writer = SummaryWriter('runs/loss_plot')

    printc("create iteratror")
    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key = lambda x: len(x.src),
        device=device
    )

    save_model_params(os.path.join(conf['weight_folder'], f"{from_lang}to{to_lang}.conf"),
        embedding_size=embedding_size,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        forward_expansion=forward_expansion,
        dropout=dropout,
        max_len=max_len,
    )
    #*load_model_params(conf_file_path).values()
    if r:
        model = Transformer(
            embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device
        ).to(device)
    else:
        model = Transformer(*load_model_params(conf_file_path).values(), device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    pad_idx = target.vocab.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    step = manager.resume(model, optimizer)
    all_loss = []


    for epoch in range(step,num_epochs):

        loop = tqdm(enumerate(train_iterator), total=len(train_iterator), leave=False)
        for batch_id, batch in loop:
            src_data = batch.src.to(device)
            trg_data = batch.trg.to(device)

            output = model(src_data, trg_data[:-1]) #
            
            output = output.reshape(-1, output.shape[2])
            trg_data = trg_data[1:].reshape(-1)
            optimizer.zero_grad()

            loss = criterion(output, trg_data)

            all_loss.append(loss.item())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

            optimizer.step()

            writer.add_scalar("training_loss", loss, global_step=step)
            step += 1

            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss = loss.item())

        mean_loss = sum(all_loss) / len(all_loss)
        scheduler.step(mean_loss)
        manager.update(model, optimizer, step)


    score = bleu(valid_data[1:100], model, from_lang, to_lang, device)
    print(f"Bleu score: {score*100:.2f}")


if __name__ =="__main__":
    translater()