
import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys
import os
import json

def printc(text):
    print(f'[\33[32m>\33[0m] {text}')


def translate_sentence(model, sentence, source_lang, target_lang, spcacy_src_lang, device, max_length=50):
    # Load german tokenizer
    #spacy_fr = spacy.load(f"{source_lang}_core_news_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spcacy_src_lang(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, source_lang.init_token)
    tokens.append(source_lang.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [source_lang.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [target_lang.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == target_lang.vocab.stoi["<eos>"]:
            break

    print(outputs)
    translated_sentence = [target_lang.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]

def bleu(data, model, source_lang, target_lang, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, source_lang, target_lang, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


class checkpointManager:
    def __init__(self, filepath):
        self.filepath = filepath
        if os.path.exists(self.filepath):
            self.state = torch.load(self.filepath)
        else:
            #if os.path.isfile(self.filepath):
            folder = os.path.dirname(self.filepath)
            print(folder)
            os.makedirs(folder, exist_ok=True)
            self.state = None

            #else:
                #raise ValueError('filepath need a file into')

    def resume(self, model, optimizer):
        if self.state is None:
            return 0 #step = 0
        else:
            model.load_state_dict(self.state["state_dict"])
            optimizer.load_state_dict(self.state["optimizer"])
            return self.state["step"]

    def update(self, model,optimizer,step):
        checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
        
        if os.path.exists(self.filepath):
            if self.state is not None and checkpoint["step"] >= self.state["step"]:
                self.state = checkpoint
                torch.save(checkpoint, self.filepath)

        else:
            self.state = checkpoint
            torch.save(checkpoint, self.filepath)

    def save_model_params(self, filename, **params):
        data = {i:j for i,j in params.items()}
        with open(filename, 'w') as f:
            json.dump(data, filename)

    def load_model_params(self, filename):
        with open(filename, 'w') as f:
            data = json.load(f)
            return data