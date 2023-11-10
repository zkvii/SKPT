import os
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
# from read_gragh import wordlist_to_keyword_vec
from src.utils import config
from torch.utils.data import Dataset
from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union, Optional
from src.utils.comet import Comet,CometMulti
from more_itertools import chunked


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")


class Lang:
    def __init__(self, init_index2word: Dict[int, str]):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence: List[str]):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def load_dataset() -> Tuple[Dict, Dict, Dict, Lang]:
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc_skpt.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    # for i in range(3):
    #     print("[situation]:", " ".join(data_tra["situation"][i]))
    #     print("[emotion]:", data_tra["emotion"][i])
    #     print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
    #     print("[target]:", " ".join(data_tra["target"][i]))
    #     print(" ")
    return data_tra, data_val, data_tst, vocab


def read_files(vocab: Lang) -> Tuple[Dict, Dict, Dict, Lang]:
    files = DATA_FILES(config.data_dir)
    
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]
    
    # test 1000
    train_files=[items[:1000] for items in train_files]
    
    data_train = build_dataset(vocab, train_files)
    data_dev = build_dataset(vocab, dev_files)
    data_test = build_dataset(vocab, test_files)

    return data_train, data_dev, data_test, vocab

def get_wordnet_pos(tag:str)->Optional[str]:
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None

class CometInference:
    _instance = CometMulti("data/Comet", config.device,config.batch_size//2)
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CometInference, cls).__new__(cls)
        return cls._instance

        

def get_commonsense(vocab:Lang,items:List[List[str]],data_dict:Dict)->None:
    # Get commonsense for each item
    comet_model=CometInference()
    # Combine items into batches to speed up inference
    input_sentences=[" ".join(item) for item in items]
    batch_input_sentences=list(chunked(input_sentences,config.batch_size//2))
    for batch_input in tqdm(batch_input_sentences):
        # Get commonsense for each relation
        for relation in relations:
            # [8*5*6] batch_cs_ret
            batch_cs_ret=comet_model.generate(batch_input,relation)
            # Process the commonsense sentences
            cs_ret=[process_sent(c) for c in cs_ret for cs_ret in batch_cs_ret]
            # Add the commonsense sentences to the vocab
            for cs in cs_ret:
                vocab.index_words(cs)
            # Add the commonsense sentences to the data dict
            data_dict[relation].append(cs_ret)
            # cs_list.append(cs_ret)
         

def build_context(vocab: Lang, contexts: List, data_dict: Dict) -> None:
    # every dialog contains one context
    rel_trigger_ctxs=[]
    for ctx in tqdm(contexts):
        ctx_list=[]
        emotion_word_list=[]
        # every context contains several sentences
        for i,c in enumerate(ctx):
            item= process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            
            ws_pos=nltk.pos_tag(item)
            for pair in ws_pos:
                word_pos=get_wordnet_pos(pair[1])
                if pair[0] not in stop_words and (word_pos==wordnet.ADJ or pair[0] in emotion_lexicon):
                    emotion_word_list.append(pair[0])
            if i==len(ctx)-1:
                rel_trigger_ctxs.append(item)
                # get commonsense for last sentence
                # get_commonsense(vocab,item,data_dict)
        
        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(emotion_word_list)
    # get commonsense for all context
    get_commonsense(vocab,rel_trigger_ctxs,data_dict)
    
def process_sent(sentence):
    """standardlize sentence using nltk word_tokenize"""
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def build_dataset(vocab: Lang, corpus: List[List[str]]) -> Dataset:
    # corpus:[array[context],array[target],array[emotion],array[situation]
    def standardize(sents: List[str]) -> List[List[str]]:
        collect_sents = []
        for item in tqdm(sents):
            item = process_sent(item)
            vocab.index_words(item)
            collect_sents.append(item)
        return collect_sents

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "xWant": [],
        "xNeed": [],
        "xIntent": [],
        "xEffect": [],
        "xReact": [],
    }

    contexts = corpus[0]
    targets = corpus[1]
    emotions = corpus[2]
    situations = corpus[3]
    build_context(vocab, contexts, data_dict)
    data_dict["emotion"] = emotions
    data_dict["situation"] = standardize(situations)
    data_dict["target"] = standardize(targets)
    
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["emotion_context"])
        == len(data_dict["xWant"])
        == len(data_dict["xNeed"])
        == len(data_dict["xIntent"])
        == len(data_dict["xEffect"])
        == len(data_dict["xAffect"])
        
    )

    return data_dict


def prepare_data_loader(batch_size: int) -> Tuple[Dataset, Dataset, Dataset, Lang, int]:
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = EMDataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = EMDataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = EMDataset(pairs_tst, vocab)

    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )


class EMDataset(Dataset):
    """custom dataset for empathetic_dialogue"""

    def __init__(self, data: Dict, vocab: Lang) -> None:
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.emo_map = emo_map
        self.SA = SentimentIntensityAnalyzer()

    """get item"""

    def __getitem__(self, index) -> Dict:
        item = {}
        item["context"] = self.data["context"][index]
        item["emotion"] = self.data["emotion"][index]
        item["situation"] = self.data["situation"][index]
        item["target"] = self.data["target"][index]
        item["emotion_context"] = self.data["emotion_context"][index]

        return item

    def __len__(self) -> int:
        assert (
            len(self.data["context"])
            == len(self.data["emotion"])
            == len(self.data["situation"])
            == len(self.data["target"])
        )
        return len(self.data["target"])


def collate_fn(data: List) -> Dict:
    def pad_sequence(sequences: List[List[int]]) -> Tuple(List[int], List[int]):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()  # padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort data by context length in descending order
    data.sort(key=lambda x: len(x["context"]), reverse=True)
    # map list[dict] to dict[list]
    item_info = {key: [d[key] for d in data] for key in data[0]}

    input_batch, input_lengths = pad_sequence(
        item_info["context"], item_info["emotion_context"]
    )
    mask_input, mask_input_lengths = pad_sequence(item_info["context_mask"])
    emotion_batch, emotion_lengths = pad_sequence(item_info["emotion_context"])

    # Target
    target_batch, target_lengths = pad_sequence(item_info["target"])

    # define return dict
    batch_data = {}
    batch_data["input_batch"] = input_batch
    batch_data["input_lengths"] = torch.LongTensor(input_lengths)

    batch_data["mask_input"] = mask_input

    batch_data["emotion_batch"] = emotion_batch

    batch_data["target_batch"] = target_batch
    batch_data["target_lengths"] = torch.LongTensor(target_lengths)

    return batch_data
