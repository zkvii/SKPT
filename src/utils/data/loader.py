import os
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from read_gragh import wordlist_to_keyword_vec
from src.utils import config
import torch.utils.data as data
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

relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
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


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item, data_dict):
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)

    data_dict["utt_cs"].append(cs_list)


def encode_ctx(vocab, items, data_dict, comet):
    for ctx in tqdm(items):
        ctx_list = []
        # possibly emotional words; mainly adj and emotion_lexicon
        e_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos
            # list of (word, pos)
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                # word is not in stop_words and is adj or in emotion_lexicon
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])
            # when iterate end then get commonsenses
            if i == len(ctx) - 1:
                get_commonsense(comet, item, data_dict)

        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, files):
    from src.utils.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("data/Comet", config.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_ctx(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            # process for target and emotion_context
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab


def load_dataset():
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc.p"
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


# 根据词性筛选词


def filter_words_by_pos(words_list):
    filtered_words_list = []
    if isinstance(words_list[0], list):
        for words in words_list:
            filtered_words = []
            for word, pos in pos_tag(words):
                pos = pos[0].upper()  # 取词性标记的首字母
                if pos in ["N", "V", "J"]:  # 名词、动词、形容词的词性标记
                    filtered_words.append(word)
            filtered_words_list.append(filtered_words)
    else:
        for word, pos in pos_tag(words_list):
            pos = pos[0].upper()  # 取词性标记的首字母
            if pos in ["N", "V", "J"]:  # 名词、动词、形容词的词性标记
                filtered_words_list.append(word)
    return filtered_words_list


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]

        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )
        (
            item["emotion_context"],
            item["emotion_context_mask"],
        ) = self.preprocess(item["emotion_context"])

        item["cs_text"] = self.data["utt_cs"][index]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        # 模型改动2：过滤非动词、名词、形容词的token*****************************************************************************
        # if config.model == "cem_en" or 'skpt':
        item["context_keywords"] = filter_words_by_pos(item["context_text"])
        item["x_intent_txt_filtered"] = filter_words_by_pos(item["x_intent_txt"])
        item["x_need_txt_filtered"] = filter_words_by_pos(item["x_need_txt"])
        item["x_want_txt_filtered"] = filter_words_by_pos(item["x_want_txt"])
        item["x_effect_txt_filtered"] = filter_words_by_pos(item["x_effect_txt"])
        item["target_keywords_text_filtered"] = filter_words_by_pos(item["target_text"])
        # word2vec keywords part
        item["x_intent_filtered"] = self.preprocess(
            item["x_intent_txt_filtered"], cs=True
        )
        item["x_need_filtered"] = self.preprocess(item["x_need_txt_filtered"], cs=True)
        item["x_want_filtered"] = self.preprocess(item["x_want_txt_filtered"], cs=True)
        item["x_effect_filtered"] = self.preprocess(
            item["x_effect_txt_filtered"], cs=True
        )
        # 模型改动2：过滤非动词、名词、形容词的token*****************************************************************************

        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")

        # if config.model == 'cem_en' or 'skpt':
        # ---------------------target_keywords---------------------
        item["senti"] = self.preprocess_senti(item["context_text"])  # 得到情感极性的值
        item["target_keywords"] = self.preprocess(
            item["target_keywords_text_filtered"], anw=True
        )  # 得到target的keywords的序号
        # 此处不能直接改原始的item["target"]，因为item["target"]在计算其他loss时有用，不能改变，因此需要新建一个item["target_keywords"]
        # ---------------------target_keywords---------------------(end)

        # -----------------------------add graph info-----------------------
        context_keywords = [
            word for words_list in item["context_keywords"] for word in words_list
        ]
        x_affect_keywords = [
            word for words_list in item["x_react_txt"] for word in words_list
        ]
        # expand keywords
        x_intent_keywords = (
            context_keywords
            + [
                word
                for words_list in item["x_intent_txt_filtered"]
                for word in words_list
            ]
            + x_affect_keywords
        )
        x_need_keywords = (
            context_keywords
            + [
                word
                for words_list in item["x_need_txt_filtered"]
                for word in words_list
            ]
            + x_affect_keywords
        )
        x_want_keywords = (
            context_keywords
            + [
                word
                for words_list in item["x_want_txt_filtered"]
                for word in words_list
            ]
            + x_affect_keywords
        )
        x_effect_keywords = (
            context_keywords
            + [
                word
                for words_list in item["x_effect_txt_filtered"]
                for word in words_list
            ]
            + x_affect_keywords
        )

        item["x_intent_keywords"] = self.preprocess(x_intent_keywords, keywords=True)
        item["x_need_keywords"] = self.preprocess(x_need_keywords, keywords=True)
        item["x_want_keywords"] = self.preprocess(x_want_keywords, keywords=True)
        item["x_effect_keywords"] = self.preprocess(x_effect_keywords, keywords=True)

        # graph matrix
        item["x_intent_keyword_matrix"] = wordlist_to_keyword_vec(x_intent_keywords)
        item["x_need_keyword_matrix"] = wordlist_to_keyword_vec(x_need_keywords)
        item["x_want_keyword_matrix"] = wordlist_to_keyword_vec(x_want_keywords)
        item["x_effect_keyword_matrix"] = wordlist_to_keyword_vec(x_effect_keywords)

        # ------------------------------add graph info----------------------(end)

        return item

    def preprocess(self, arr, anw=False, cs=None, emo=False, keywords=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            sequence = [config.CLS_idx] if cs != "react" else []
            for sent in arr:
                sequence += [
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["to", "none"]
                ]

            return torch.LongTensor(sequence)
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)
        elif keywords:
            # temp solution for unseen word
            self.vocab.index_words(arr)
            # donot add clstoken
            sequence = [self.vocab.word2index[word] for word in arr]
            return torch.LongTensor(sequence)
        else:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask)

    def preprocess_senti(self, arr):  # 根据情感词典计算情感极性值
        x_senti = [0]
        for i, sentence in enumerate(arr):
            x_senti.extend(replace_tokens_with_sentiments(sentence))
        return torch.LongTensor(x_senti)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


# 得到一个单词的情感值


def get_sentiment_scores(word):
    synsets = list(swn.senti_synsets(word))
    if synsets:
        synset = synsets[0]
        pos_score = synset.pos_score()
        neg_score = synset.neg_score()
        obj_score = synset.obj_score()
        return pos_score, neg_score, obj_score
    else:
        return None, None, None


# 得到一个token列表的情感值，如果词在词典中不存在，情感值为0


def replace_tokens_with_sentiments(token_list):
    new_values = []
    for token in token_list:
        pos_score, neg_score, obj_score = get_sentiment_scores(token)
        if pos_score is not None:
            max_score = max(pos_score, neg_score, obj_score)
            if max_score == pos_score:
                new_value = pos_score * 10
            elif max_score == neg_score:
                new_value = (neg_score / 3.0) * 10
            else:
                new_value = 0
        else:
            new_value = 0
        new_values.append(new_value)
    return new_values


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()  # padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_senti(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(
            len(sequences), max(lengths)
        ).long()  # padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # ------------------ pad stack matrix ------------

    def pad_matrix(sequences):
        max_size = max([tensor.size(0) for tensor in sequences])
        padded_tensors = [
            F.pad(t, (0, max_size - t.shape[0], 0, max_size - t.shape[0]), value=1)
            for t in sequences
        ]
        stacked_tensor = torch.stack(padded_tensors)
        return stacked_tensor

    # -----------------pad stack matrix--------------(end)

    # def pad_keywords(sequences):

    data.sort(key=lambda x: len(x["context"]), reverse=True)  # sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])

    # Target
    target_batch, target_lengths = merge(item_info["target"])

    # --------------------merge senti and keywords--------------
    # if config.model == 'cem_en' or 'skpt':
    senti_batch, senti_length = merge_senti(item_info["senti"])
    senti_batch = senti_batch.to(config.device)
    target_keywords_batch, target_keywords_length = merge(item_info["target_keywords"])
    target_keywords_batch = target_keywords_batch.to(config.device)
    # --------------------merge senti and keywords--------------(end)

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    target_batch = target_batch.to(config.device)

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = emotion_batch.to(config.device)
    # --------------------------collate senti and keywords---------------------
    # if config.model == 'cem_en' or 'skpt':
    d["senti_batch"] = senti_batch
    d["target_keywords_batch"] = target_keywords_batch
    d["target_keywords_length"] = torch.LongTensor(target_keywords_length)

    d["x_intent_matrix"] = pad_matrix(item_info["x_intent_keyword_matrix"])
    d["x_need_matrix"] = pad_matrix(item_info["x_need_keyword_matrix"])
    d["x_want_matrix"] = pad_matrix(item_info["x_want_keyword_matrix"])
    d["x_effect_matrix"] = pad_matrix(item_info["x_effect_keyword_matrix"])

    d["x_intent_matrix"] = d["x_intent_matrix"].to(config.device)
    d["x_need_matrix"] = d["x_need_matrix"].to(config.device)
    d["x_want_matrix"] = d["x_want_matrix"].to(config.device)
    d["x_effect_matrix"] = d["x_effect_matrix"].to(config.device)

    # vectorize context+cog+aff
    d["x_intent_keywords"], d["x_intent_keywords_lengths"] = merge(
        item_info["x_intent_keywords"]
    )
    d["x_need_keywords"], d["x_need_keywords_lengths"] = merge(
        item_info["x_need_keywords"]
    )
    d["x_want_keywords"], d["x_want_keywords_lengths"] = merge(
        item_info["x_want_keywords"]
    )
    d["x_effect_keywords"], d["x_effect_keywords_lengths"] = merge(
        item_info["x_effect_keywords"]
    )

    d["x_intent_keywords"] = d["x_intent_keywords"].to(config.device)
    d["x_need_keywords"] = d["x_need_keywords"].to(config.device)
    d["x_want_keywords"] = d["x_want_keywords"].to(config.device)
    d["x_effect_keywords"] = d["x_effect_keywords"].to(config.device)

    # --------------------------collate senti and keywords---------------------(end)
    # program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    # text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
    for r in relations:
        pad_batch, _ = merge(item_info[r])
        pad_batch = pad_batch.to(config.device)
        d[r] = pad_batch
        d[f"{r}_txt"] = item_info[f"{r}_txt"]

    return d


def prepare_data_seq(batch_size: int):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)

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
