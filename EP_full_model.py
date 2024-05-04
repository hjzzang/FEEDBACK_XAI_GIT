#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : model.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/17 14:57
@version: 1.0
@desc  :
"""
import torch

from transformers import RobertaConfig, RobertaModel
from torch import nn

class ExplainableModel(nn.Module):
    def __init__(self, bert_dir):
        super().__init__()
        #self.bert_dir = r'D:\2022_xai\roberta_base'
        self.bert_config = RobertaConfig.from_pretrained(bert_dir, output_hidden_states=False)
        self.intermediate = RobertaModel.from_pretrained(bert_dir)

        self.span_info_collect = SICModel(self.bert_config.hidden_size)
        self.interpretation = InterpretationModel(self.bert_config.hidden_size)
        #self.tok_size = int(512)
        self.tok_size = int(512/2)
        self.output_pre = nn.Linear(self.bert_config.hidden_size, 1)
        self.output = nn.Linear(self.tok_size, 2)


    def forward(self, citing_tok_ids, cited_tok_ids, citing_cs_idx, cited_cs_idx):

        # intermediate layer
        citing_tok_ids_ = citing_tok_ids.reshape(-1, citing_tok_ids.shape[2])
        cited_tok_ids_ = cited_tok_ids.reshape(-1, cited_tok_ids.shape[2])
        # max_(citing_cs + cited_cs) -> 34!!!!
        citing_attention_mask = (citing_tok_ids_ != 1).long()
        cited_attention_mask = (cited_tok_ids_ != 1).long()
        citing_emb, citing_start = self.intermediate(citing_tok_ids_.long(),
                                                attention_mask=citing_attention_mask)  # torch.Size([28, 512, 768])
        cited_emb, cited_start = self.intermediate(cited_tok_ids_.long(),
                                              attention_mask=cited_attention_mask)  # torch.Size([40, 512, 768])


        citing_emb = citing_emb.reshape(citing_tok_ids.shape[0], -1, citing_emb.shape[1],
                                        citing_emb.shape[2])  # torch.Size([2, 14, 512, 768])
        cited_emb = cited_emb.reshape(cited_tok_ids.shape[0], -1, cited_emb.shape[1],
                                      cited_emb.shape[2])  # torch.Size([2, 20, 512, 768])


        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(citing_emb, cited_emb, citing_cs_idx, cited_cs_idx)
        #print("h_ij:", str(h_ij.shape))
        # interpretation layer
        H, a_ij = self.interpretation(h_ij, citing_cs_idx)
        #print("H:", str(H.shape))
        #print("a_ij:", str(a_ij.shape))
        #print("a_ij:", str(a_ij))
        # output layer
        out_pre = self.output_pre(H).reshape(H.shape[0], H.shape[1])
        #print("out_pre:", str(out_pre.shape))
        out = self.output(out_pre)
        #print("out:", str(out.shape))

        return out, a_ij

class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        # self.W_3 = nn.Linear(hidden_size, hidden_size)
        # self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, citing_emb, cited_emb, citing_cs_idx, cited_cs_idx):
        # W_1 = nn.Linear(hidden_size, hidden_size).to(torch.device("cuda:0"))

        #W1_citing_h = self.W_1(citing_emb)  # torch.Size([2, 14, 512, 768])
        #W1_cited_h = self.W_1(cited_emb)  # torch.Size([2, 20, 512, 768])
        W2_citing_h = self.W_2(citing_emb) # torch.Size([2, 20, 512, 768])
        W2_cited_h = self.W_2(cited_emb)  # torch.Size([2, 20, 512, 768])
        #print("W1_citing_h:", str(W1_citing_h.shape))
        #print("W1_cited_h:", str(W1_cited_h.shape))
        #print("W2_citing_h:", str(W2_citing_h.shape))
        #print("W2_cited_h:", str(W2_cited_h.shape))

        # span = (W1_h_citing_emb + W2_h_cited_emb + (W3_h_citing_emb - W3_h_cited_emb) + torch.mul(W4_h_citing_emb,W4_h_cited_emb))
        #span = (torch.index_select(W1_citing_h, 1, citing_cs_idx)+torch.index_select(W1_cited_h, 1, cited_cs_idx)+torch.index_select(W2_citing_h, 1, citing_cs_idx) - torch.index_select(W2_cited_h, 1, cited_cs_idx))
        span = (torch.index_select(W2_citing_h, 1, citing_cs_idx) - torch.index_select(W2_cited_h, 1, cited_cs_idx))
        #print("span:", str(span.shape))
        h_ij = torch.tanh(span)  # torch.Size([2, 280, 512, 768])
        #print("h_ij:", str(h_ij.shape))
        return h_ij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h1_t = nn.Linear(hidden_size, 1)
        #self.h_t = nn.Linear(hidden_size, 1).to(torch.device("cuda:0"))
        self.tok_size = int(512/2)
        #self.tok_size = int(200)
        self.h2_t = nn.Linear(self.tok_size,1)

    def forward(self, h_ij, citing_cs_idx):
        #h_t = nn.Linear(hidden_size, 1).)("cuda:0"))
        o_ij = self.h1_t(h_ij).squeeze(-1)  # (ba, span_num)
        # o_ij = h_t(h_ij).squeeze(-1)  # (ba, span_num)
        # mask illegal span
        o_ij_2 = self.h2_t(o_ij).squeeze(-1) #torch.Size([2, 280])
        o_ij_2 = o_ij_2 - citing_cs_idx
        #print("o_ij_2 :", str(o_ij_2.shape))
        # normalize all a_ij, a_ij sum = 1
        a_ij = nn.functional.softmax(o_ij_2, dim=1)
        #print("a_ij :", str(a_ij.shape))
        # weight average span representation to get H
        H = (a_ij.unsqueeze(-1).unsqueeze(-1) * h_ij).sum(dim=1)  # (bs, tok_size, hidden_size)

        return H, a_ij

"""
bert_path = r'E:\PycharmProjects\2022_xai\roberta_base'
bert_dir = r'E:\PycharmProjects\2022_xai\roberta_base'
model = ExplainableModel(bert_path)

output = model(citing_tok_ids, cited_tok_ids,citing_cs_idx, cited_cs_idx)
"""
