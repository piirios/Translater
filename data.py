from torchtext.data import Field, BucketIterator, TabularDataset
import os
import pandas as pd
import spacy

BASE_FILENAME = "small_vocab_{}.csv"
CONCAT_FILENAME = "{}2{}_{}.csv"
DATASET_FOLDER = os.path.join("dataset", "v1.1")
SEP = "/////"
TEMP_SEP="µ"
INIT_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"
MAX_SIZE_VOCAB = 100000
MIN_FREQ_VOCAB = 2

def get_data_root(folder, from_lang, to_lang, type_of_learn):
    """
    from_lang_root = f"/{from_lang}/{to_lang}/{type_of_learn}/segments.{from_lang}"
    to_lang_root = f"/{from_lang}/{to_lang}/{type_of_learn}/segments.{to_lang}"
    """
    from_lang_root = os.path.join(from_lang, to_lang, type_of_learn, f"segments.{from_lang}")
    to_lang_root = os.path.join(from_lang, to_lang, type_of_learn, f"segments.{to_lang}")
    return os.path.join(folder, from_lang_root), os.path.join(folder, to_lang_root)


def tokenize(spacy_lang):
    def tokenize_fr(text):
        return [tok.text for tok in spacy_lang.tokenizer(text)]

def get_data(src_lang, trg_lang, dataset_folder, type_of_learn):
        fromlang, tolang = get_data_root(dataset_folder, src_lang, trg_lang, type_of_learn)
        #on vérifie si le si le fichier des deux datasets concaténée exste
        if not os.path.exists(os.path.join(dataset_folder, CONCAT_FILENAME.format(src_lang, trg_lang, type_of_learn))):
            src_df = pd.read_csv(fromlang, sep=SEP, encoding='utf-8') #on charge les fichiers csv des langues
            trg_df = pd.read_csv(tolang, sep=SEP, encoding='utf-8')
            dataset_df = pd.concat([src_df, trg_df], axis=1) #on les ajoutent sur l'axe 1(l'axe des colonnes)
            dataset_df.to_csv(os.path.join(dataset_folder, CONCAT_FILENAME.format(src_lang, trg_lang, type_of_learn)),sep=TEMP_SEP, encoding='utf-8-sig', index = False) #on enregistre le fichier
        
        spacy_src = spacy.load(f"{src_lang}_core_news_sm")
        spacy_trg = spacy.load(f"{trg_lang}_core_web_sm")

        source = Field(tokenize=tokenize(spacy_src), lower=True, init_token=INIT_TOKEN, eos_token=END_TOKEN)
        target = Field(tokenize=tokenize(spacy_trg), lower=True, init_token=INIT_TOKEN, eos_token=END_TOKEN)

        dataset = TabularDataset(os.path.join(dataset_folder, CONCAT_FILENAME.format(src_lang, trg_lang, type_of_learn)), format="csv", fields=[("src", source), ("trg", target)], csv_reader_params={'delimiter':TEMP_SEP})

        train_data, valid_data = dataset.split()

        source.build_vocab(train_data, max_size=MAX_SIZE_VOCAB, min_freq=MIN_FREQ_VOCAB)
        target.build_vocab(train_data, max_size=MAX_SIZE_VOCAB, min_freq=MIN_FREQ_VOCAB)


        return train_data, valid_data, source, target, spacy_src, spacy_trg

    
if __name__ == "__main__":
    fromlang, tolang = get_data_root(DATASET_FOLDER, "fr", 'en', 'train')
    data = get_data("fr", "en", DATASET_FOLDER, "train")
