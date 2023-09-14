# TAKEN FROM https://github.com/kolloldas/torchnlp
import os
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
import nltk
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import re

import numpy as np
import math
from src.models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    get_keywords_output_from_batch,
    top_k_top_p_filtering,
)
from src.utils import config
from src.utils.constants import MAP_EMO
if config.model == "skpt":
    from src.utils.decode.cem import Translator

from sklearn.metrics import accuracy_score


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            # for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(
            embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params)
                                     for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:,
                                    : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            # for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(
            embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:,
                                 : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:,
                                        : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:,
                                    : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec(
                (x, encoder_output, [], (src_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  # extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  # extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 4 if config.woEMO else 5
        input_dim = input_num * config.hidden_dim
        hid_num = 2 if config.woEMO else 3
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x

# 自定义损失函数


class EmoMLP(nn.Module):
    def __init__(self,decoder_num=32):
        super().__init__()
        
        self.emo_lin_1 = nn.Linear(config.hidden_dim*5, config.hidden_dim*2)
        self.emo_lin_2 = nn.Linear(config.hidden_dim*2, decoder_num)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.emo_lin_1(x)
        x = self.act(x)
        x = self.emo_lin_2(x)
        return x


class CustomLoss(nn.Module):
    def __init__(self, vocab):
        super(CustomLoss, self).__init__()
        self.vocab = vocab
        self.tfidf_vectorizer = TfidfVectorizer()

    def forward(self, predicted, target):
        loss = 0.0
        for pred_seq, target_seq in zip(predicted, target):

            cleaned_pred = self.clean_sentence(pred_seq)
            cleaned_target = self.clean_sentence(target_seq)
            # 根据词性过滤和选择 TF-IDF 前四个词
            filtered_pred = self.filter_and_select_words(cleaned_pred)
            filtered_target = self.filter_and_select_words(cleaned_target)
            target_length = len(filtered_target)
            if len(filtered_pred) < target_length:
                filtered_pred += [''] * (target_length - len(filtered_pred))
            else:
                filtered_pred = filtered_pred[:target_length]
            # 将单词转换为索引
            pred_indices = [self.vocab.word2index[word]
                            if word in self.vocab.word2index
                            else config.UNK_idx
                            for word in filtered_pred]
            target_indices = [self.vocab.word2index[word]
                              if word in self.vocab.word2index
                              else config.UNK_idx
                              for word in filtered_target]

            # 转换为张量
            pred_indices = torch.tensor(pred_indices, dtype=torch.float)
            target_indices = torch.tensor(target_indices, dtype=torch.float)

            # 计算交叉熵损失
            criterion = nn.CrossEntropyLoss()
            # mse_loss = nn.MSELoss()
            # nll_loss = nn.NLLLoss()
            predicted_probs = nn.functional.softmax(pred_indices, dim=0)
            target_probs = nn.functional.softmax(target_indices, dim=0)

            loss += criterion(predicted_probs, target_probs)

        return loss / len(predicted)

    def clean_sentence(self, sentence):
        # 去除标点符号和非法字符
        cleaned_sentence = re.sub(r'[^\w\s]', '', sentence)
        return cleaned_sentence

    def filter_and_select_words(self, words):
        # 对文本进行过滤，只保留动词、名词和形容词
        filtered_words = []
        for word, pos in pos_tag(words.split()):
            pos = pos[0].upper()  # 取词性标记的首字母
            if pos in ['N', 'V', 'J', 'P'] and word in self.vocab.word2index:  # 名词、动词、形容词的词性标记
                filtered_words.append(word)

        """# 将单词转换为字符串
        text = ' '.join(filtered_words)
        if text.replace(' ','') != '':
            # 计算TF-IDF矩阵
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            except ValueError:
                print(text)
            
            # 提取TF-IDF值并排序
            tfidf_values = tfidf_matrix.toarray()[0]
            sorted_indices = tfidf_values.argsort()[::-1]
            
            # 保留前四个词（或所有词，如果不足四个）
            if len(sorted_indices) >= 4:
                selected_words = [filtered_words[i] for i in sorted_indices[:4]]
            elif len(sorted_indices) >=1:
                selected_words = [filtered_words[i] for i in sorted_indices[:len(sorted_indices)]]
            else:
                if len(filtered_words)>=4:
                    selected_words =filtered_words[:4]
                else:
                    selected_words = []
            while len(selected_words) < 4:
                selected_words.append('')
        else:
            selected_words=['','','','']"""

        selected_words = filtered_words
        return selected_words

# ------------------gcn module---------------


class TwoLayerGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerGCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        # Compute the normalized adjacency matrix
        degree_matrix = torch.sum(adjacency_matrix, dim=2, keepdim=True)
        normalized_adjacency_matrix = adjacency_matrix / \
            (degree_matrix + 1e-6)  # Adding a small epsilon for stability

        # Perform the first graph convolution
        print(normalized_adjacency_matrix.shape)
        print(x.shape)

        x = torch.matmul(normalized_adjacency_matrix, x)
        x = self.fc1(x)
        x = F.relu(x)

        # Perform the second graph convolution
        x = torch.matmul(normalized_adjacency_matrix, x)
        x = self.fc2(x)
        x = F.relu(x)

        return x
# ----------------gcn module----------------(end)

# ----------------gru module----------------


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through GRU
        out, _ = self.gru(x)

        # Get the output from the last time step
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)
        return out
# ----------------gru module----------------(end)

# decoder_num is emotion label number
class SKPT(nn.Module):
    def __init__(
        self,
        vocab,
        decoder_number,
        model_file_path=None,
        is_eval=False,
        load_optim=False,
        gcn_layer_num=2
    ):
        super(SKPT, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.word_freq = np.zeros(self.vocab_size)

        self.is_eval = is_eval
        self.rels = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.encoder = self.make_encoder(config.emb_dim)
        
        self.emo_encoder = self.make_encoder(config.emb_dim)
        # self.cog_encoder = self.make_encoder(config.emb_dim)

        self.senti_encoder = self.make_encoder(config.emb_dim)  # 新增的encoder

        # skpt encoder
        self.share_cog_graph_encoder = self.make_encoder(config.emb_dim)

        # self.emo_ref_encoder = self.make_encoder(2 * config.emb_dim)
        self.cog_ref_encoder = self.make_encoder(2 * config.emb_dim)

        # emo mlp
        self.emo_mlp=EmoMLP()
        # two layer gcn
        self.gcn_layer_num = gcn_layer_num
        self.KGCN = TwoLayerGCN(
            config.hidden_dim, config.hidden_dim, config.hidden_dim)

        self.KGRU = GRUModel(
            config.hidden_dim, config.hidden_dim, 1, config.hidden_dim)

        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.keywords_decoder = Decoder(  # 新的decoder
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.emo_lin = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        if not config.woCOG:
            self.cog_lin = MLP()

        self.generator = Generator(config.hidden_dim, self.vocab_size)

        self.keywords_generator = Generator(
            config.hidden_dim, self.vocab_size)  # 新的generator

        self.activation = nn.Softmax(dim=1)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(
            ignore_index=config.PAD_idx, reduction="sum")
        if not config.woDiv:
            self.criterion.weight = torch.ones(self.vocab_size)
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0,
                                 betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

        self.new_loss_fn = CustomLoss(self.vocab)

    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "skpt_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if config.EOS_idx in pred:
                ind = pred.index(config.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == config.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)

    def forward(self, batch):
        enc_batch = batch["input_batch"]  # batch_size*92 （92为seq length）
        src_mask = enc_batch.data.eq(
            config.PAD_idx).unsqueeze(1)  # batch_size*1*92
        senti_emb = self.embedding(batch['senti_batch'])  # 新增的senti embedding
        senti_outputs = self.senti_encoder(senti_emb, src_mask)
        # batch_size*92*300（300为embedding dim）
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb + \
            senti_emb  # batch_size*92*300
        # batch_size * seq_len * 300 (#batch_size*92*300)
        # concat pool graph
        enc_outputs = self.encoder(src_emb, src_mask)
        dim=[-1,enc_outputs.shape[1],-1]
        
        # -----------------------compute_keyword_graph----------------

        graph_cogs = []
        cog_encs=[]
        for r in self.rels:
            if r != "x_react":
                cog_batch = batch[r+'_keywords']
                cog_mask = cog_batch.data.eq(config.PAD_idx).unsqueeze(1)
                cog_emb = self.embedding(cog_batch)

                cog_adj = batch[r+'_matrix']
                
                graph_cog_enc = self.share_cog_graph_encoder(
                    cog_emb, cog_mask)
        
                
                cog_encs.append(graph_cog_enc)
                cog_gcn_output = self.KGCN(graph_cog_enc, cog_adj)
                graph_cogs.append(cog_gcn_output)

        # batch_size * 4_seq_len * 300
        pooled_graph_cogs = torch.cat(graph_cogs, dim=1)

        if not config.woS:
            senti_extend=[torch.mean(graph_output,dim=1) for graph_output in graph_cogs] +[torch.mean(senti_outputs,dim=1)] 
            senti_concat= torch.cat(senti_extend,dim=1)

            # mean
            # senti_cls_exp=torch.mean(senti_concat,dim=1).unsqueeze(1)
            
            emo_logits=self.emo_mlp(senti_concat)
            
            
        else:
            emo_logits=self.emo_lin(enc_outputs[:,0])
            
            
        if not config.woKPT:
            # add to dim 1
            # dim=[-1,pooled_graph_cogs.shape[1],-1]
            senti_outputs_cls=torch.mean(senti_outputs,dim=1).unsqueeze(1)
            # senti_out_expand=senti_outputs_cls.expand(dim)
            # res connect
            for i in range(len(cog_encs)):
                cog_encs[i][:,0]=cog_encs[i][:,0]+senti_outputs_cls.squeeze(1)
                cog_encs[i]+=graph_cogs[i]
                cog_encs[i]=self.KGRU(cog_encs[i])
                cog_cls=cog_encs[i][:,0].unsqueeze(1).expand(dim)
                
                enc_outputs=torch.cat([enc_outputs,cog_cls],dim=2)
                # enc_outputs=torch.cat([enc_outputs,cog_encs[i][:0]],dim=1)
            # batch_size*seq_len*(5*300)
            enc_outputs=self.cog_lin(enc_outputs)
            keywords_pred_logits=torch.concat(cog_encs,dim=1)

        # dim = [-1, enc_outputs.shape[1], -1]  # 保持第一和第三维度不变，改变第二个维度
        if not config.woKPT:
            return src_mask, enc_outputs, emo_logits,keywords_pred_logits
        # return None, None, emo_logits
        return src_mask, enc_outputs, emo_logits
            
            
            
        
        # sentiment sent_enc + pooled_graph_cogs
        

        # -----------------------compute_keyword_graph----------------(end)

        # 5个，每个都是batchsize*1*300
        # cls_tokens = [c[:, 0].unsqueeze(1) for c in cs_outputs]

        # # Shape: batch_size * 1 * 300
        # cog_cls = cls_tokens[:-1]  # 4个，包含前四个cls_tokens的值
        # # batch_size * 1 * 300
        # emo_cls = torch.mean(cs_outputs[-1], dim=1).unsqueeze(1)

        # # 模型改动3：从senti_encoder得到的senti output，将结果取cls然后与emo output相加****************************************************
        # senti_cls = torch.mean(senti_outputs, dim=1).unsqueeze(
        #     1)  # batch_size * 1 * 300
        # emo_cls += senti_cls  # 把senti的cls和emo的cls相加
        # # 模型改动3：从senti_encoder得到的senti output，将结果取cls然后与emo output相加****************************************************


        # return src_mask, cog_ref_ctx, emo_logits
        # Commonsense relations
        # cs_embs = []
        # cs_masks = []
        # cs_outputs = []
        # for r in self.rels:
        #     if r != "x_react":
        #         # 模型改动1：把x_react直接连接在其他4个后面*****************************************************************************
        #         # 把x_react直接连接在其他4个后面，构造出新的输入
        #         new_batch = torch.cat((batch[r], batch['x_react']), dim=1)
        #         emb = self.embedding(new_batch).to(config.device)
        #         # emb = self.embedding(batch[r]).to(config.device)
        #         mask = new_batch.data.eq(
        #             config.PAD_idx).unsqueeze(1)  # batchsize*1*x
        #         # mask = batch[r].data.eq(config.PAD_idx).unsqueeze(1)
        #         enc_output = self.cog_encoder(emb, mask)  # batchsize*x*300
        #         # 模型改动1：把x_react直接连接在其他4个后面*****************************************************************************
        #     else:
        #         # batchsize*x*300(x是commonsense序列的长度，在5个rels里不同)
        #         emb = self.embedding(batch[r]).to(config.device)
        #         mask = batch[r].data.eq(
        #             config.PAD_idx).unsqueeze(1)  # batchsize*1*x
        #         enc_output = self.emo_encoder(emb, mask)  # batchsize*x*300
        #     # cs_embs.append(emb)
        #     # cs_masks.append(mask)
        #     cs_outputs.append(enc_output)

        # Emotion
        # if not config.woEMO:
        #     emo_concat = torch.cat(
        #         [enc_outputs, emo_cls.expand(dim)], dim=-1)  # batch_size*92*600
        #     emo_ref_ctx = self.emo_ref_encoder(
        #         emo_concat, src_mask)  # batch_size*92*300
        #     emo_logits = self.emo_lin(emo_ref_ctx[:, 0])  # batch_size*32
        # else:
        #     emo_logits = self.emo_lin(enc_outputs[:, 0])

        # # Cognition
        # cog_outputs = []
        # for cls in cog_cls:
        #     cog_concat = torch.cat(
        #         [enc_outputs, cls.expand(dim)], dim=-1)  # batch_size*92*600
        #     cog_concat_enc = self.cog_ref_encoder(
        #         cog_concat, src_mask)  # batch_size*92*300
        #     cog_outputs.append(cog_concat_enc)  # 因为Cognition有四个，所以append

        # if config.woCOG:
        #     cog_ref_ctx = emo_ref_ctx
        # else:
        #     if config.woEMO:
        #         cog_ref_ctx = torch.cat(cog_outputs, dim=-1)
        #     else:
        #         cog_ref_ctx = torch.cat(
        #             cog_outputs + [emo_ref_ctx], dim=-1)  # batch_size*92*1500
        #     cog_contrib = nn.Sigmoid()(cog_ref_ctx)  # batch_size*92*1500
        #     cog_ref_ctx = cog_contrib * cog_ref_ctx
        #     cog_ref_ctx = self.cog_lin(cog_ref_ctx)  # batch_size*92*300

    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        dec_keywords_batch, _, _, _, _ = get_keywords_output_from_batch(
            batch)  # 提取了keywords的dec batch

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        if not config.woKPT:
            src_mask, ctx_output, emo_logits,keywords_pred_logits = self.forward(batch)
        else:
            src_mask, ctx_output, emo_logits = self.forward(batch)

        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            .to(config.device)
        )
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        if not config.woKPT:
            dec_keywords_batch_shift = torch.cat(
                (sos_token, dec_keywords_batch[:, :-1]), dim=1)  # 新增的keywords_batch_shift
            mask_keywords_trg = dec_keywords_batch_shift.data.eq(
                config.PAD_idx).unsqueeze(1)  # 新增的mask_keywords_trg

        # batch_size * seq_len * 300 (GloVe)
        dec_emb = self.embedding(dec_batch_shift)
        pre_logit, attn_dist = self.decoder(
            dec_emb, ctx_output, (src_mask, mask_trg))

        # 新增的dec_keywords_emb，作为新增的keywords_decoder的输入
        if not config.woKPT:
            dec_keywords_emb = self.embedding(dec_keywords_batch_shift)
            pre_keywords_logit, attn_keywords_dist = self.keywords_decoder(
                dec_keywords_emb, ctx_output, (src_mask, mask_keywords_trg))

        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )
        if not config.woKPT:
            keywords_logit = self.keywords_generator(  # 新增的keywords_generator得到新的keywords_logit
                pre_keywords_logit,
                attn_keywords_dist,
                enc_batch_extend_vocab if config.pointer_gen else None,
                extra_zeros,
                attn_dist_db=None,
            )

        emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
        emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_label).to(config.device)
        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        keyword_loss=0
        # 模型改动4：新增keyword_loss******************************************************************
        if not config.woKPT:
            keyword_loss = self.criterion_ppl(
                keywords_logit.contiguous().view(-1, logit.size(-1)),
                dec_keywords_batch.contiguous().view(-1),
            )
        # 模型改动4：新增keyword_loss******************************************************************

        if not (config.woDiv):
            _, preds = logit.max(dim=-1)
            preds = self.clean_preds(preds)
            self.update_frequency(preds)
            self.criterion.weight = self.calc_weight()
            not_pad = dec_batch.ne(config.PAD_idx)
            target_tokens = not_pad.long().sum().item()
            div_loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )
            div_loss /= target_tokens
            # 模型改动5：给四个loss设定不同权重******************************************************************
            weight = [[1, 1.5, 1, 0.5], [1, 1.5, 1, 1.5], [1, 1, 1, 1], [0.5, 0.75, 0.5, 2],
                      [1, 0.75, 1, 1], [1, 0.75, 1, 0.5], [
                          1, 0.75, 1, 0.75], [1, 0.5, 1, 0.5],
                      [1, 0.5, 1.5, 0.5]]  # 四个loss的不同权重
            weight_set = 8  # 选用哪一套权重，当此值为-1时为初始权重（目前weight_set取2和6时效果较好）
            if weight_set >= 0 and weight_set < len(weight):
                loss = weight[weight_set][0]*emo_loss + weight[weight_set][1]*div_loss + \
                    weight[weight_set][2]*ctx_loss + \
                    weight[weight_set][3]*keyword_loss  # 新增loss
            else:
                loss = emo_loss + 1.5 * div_loss + ctx_loss + keyword_loss  # 新增loss
            # 模型改动5：给四个loss设定不同权重******************************************************************
        else:
            loss = emo_loss + ctx_loss

        pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        # print results for testing
        top_preds = ""
        comet_res = {}

        if self.is_eval:
            top_preds = emo_logits.detach().cpu().numpy().argsort()[
                0][-3:][::-1]
            top_preds = f"{', '.join([MAP_EMO[pred.item()] for pred in top_preds])}"
            for r in self.rels:
                txt = [[" ".join(t) for t in tm]
                       for tm in batch[f"{r}_txt"]][0]
                comet_res[r] = txt

        if train:
            loss.backward()
            self.optimizer.step()

        return (
            ctx_loss.item(),
            math.exp(min(ctx_loss.item(), 100)),
            emo_loss.item(),
            program_acc,
            top_preds,
            comet_res,
        )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            _,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_output, (src_mask, mask_trg)
                )

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(
                    next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(
                    next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent
