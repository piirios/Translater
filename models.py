import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, checkpointManager, printc
#from torchtext.datasets import WMT14
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import warnings
warnings.filterwarnings(action="ignore")

"""
spacy_fr = spacy.load("fr_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

french = Field(tokenize=tokenize_fr, lower=True, init_token="<SOS>", eos_token="<EOS>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<SOS>", eos_token="<EOS>")

printc("loading and spliting dataset")
train_data, valid_data, test_data = WMT14.splits(
    exts=(".de", ".en"),
    fields=(german, english) 
)

dataset = TabularDataset("dataset/eng-french/fr2en.csv", format="csv", fields=[("french", french), ("english", english)], csv_reader_params={'delimiter':':'})
#train_data, valid_data, test_data = dataset.split()
train_data, valid_data = dataset.split()



printc("build french vocab") 
french.build_vocab(train_data, max_size=10000, min_freq=2)
printc("build english vocab")
english.build_vocab(train_data, max_size=10000, min_freq=2)

"""

class Transformer(nn.Module):
    def __init__(
        self,
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
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out



if __name__ == "__main__":
    manager = checkpointManager('/home/louis/Bureau/dev/Translater/train/checkpoint.pth')

    #hyperparams
    num_epochs = 20
    lr = 3e-4
    batch_size = 32
    src_vocab_size = len(french.vocab)
    trg_vocab_size = len(english.vocab)
    embedding_size = 1024
    num_heads = 6
    num_encoder_layers = 6
    num_encoder_layers = 6
    dropout = 0.1
    max_len = 100
    forward_expansion = 2048
    src_pad_idx = english.vocab.stoi["<PAD>"]

    writer = SummaryWriter('runs/loss_plot')

    printc("create iteratror")
    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key = lambda x: len(x.french),
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

    pad_idx = french.vocab.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    step = manager.resume(model, optimizer)
    all_loss = []


    for epoch in range(num_epochs):

        loop = tqdm(enumerate(train_iterator), total=len(train_iterator), leave=False)
        for batch_id, batch in loop:
            src_data = batch.french.to(device)
            trg_data = batch.english.to(device)

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


    score = bleu(valid_data[1:100], model, french, english, device)
    print(f"Bleu score: {score*100:.2f}")