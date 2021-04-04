import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, checkpointManager, printc
#from torchtext.datasets import WMT14
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models import Transformer
from data import get_data

#hyperparams
num_epochs = 20
lr = 3e-4
batch_size = 32
embedding_size = 1024
num_heads = 6
num_encoder_layers = 6
num_encoder_layers = 6
dropout = 0.1
max_len = 100
forward_expansion = 2048

DATASET_FOLDER="dataset/v1.1"

def train(src, trg, dataset_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, valid_data, source, target, spacy_src, spacy_trg = get_data("fr", "en", DATASET_FOLDER, "train")
    manager = checkpointManager('/train/checkpoint.pth.tar')


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

    model = Transformer(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_encoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

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

            manager.update(model, optimizer, step)

            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss = loss.item())

        mean_loss = sum(all_loss) / len(all_loss)
        scheduler.step(mean_loss)


    score = bleu(valid_data[1:100], model, src, trg, device)
    print(f"Bleu score: {score*100:.2f}")
    return model

if __name__ == "__main__":
    translater = train("fr", "en")