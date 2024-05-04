import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pandas as pd


class EPfulldataset(Dataset):
    def __init__(self, directory, prefix, bert_path, max_length: int = int(512/2)):
        super().__init__()
        self.max_length = max_length
        # self.max_cc_len = 7676
        srprts = pd.read_json(directory + prefix + '.json', orient='records')

        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)
        self.result = []
        for hj in range(len(srprts)):
            line = srprts.iloc[hj]
            self.result.append((line['pair_srprt'], line['label']))

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        # pair_id, pair_srprt, label = self.result[idx]
        pair_srprt, label = self.result[idx]

        citing_tok_ids = torch.FloatTensor(
            [self.tokenizer.encode(i, add_special_tokens=False, padding='max_length', max_length=self.max_length,
                                   truncation=True) for i in pair_srprt[0]])
        cited_tok_ids = torch.FloatTensor(
            [self.tokenizer.encode(i, add_special_tokens=False, padding='max_length', max_length=self.max_length,
                                   truncation=True) for i in pair_srprt[1]])

        return citing_tok_ids, cited_tok_ids, torch.LongTensor([label])

#test
"""
from functools import partial
bert_path = r'D:\2022_xai\roberta_base'
data_dir = r'data/pair_df_small_'
prefix ='train'

import pandas as pd
#full_raw_df = pd.read_json('data/pair_df_test.json')
dataset = EPfulldataset(directory=data_dir, prefix=prefix, bert_path=bert_path)

from EP_full_collate_functions import collate_input

dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=partial(collate_input, fill_values=[1,1,0]),
            shuffle=False,
            drop_last=False
        )

i=0

for citing_tok_ids, cited_tok_ids, labels, citing_cs_idx, cited_cs_idx, max_lengths in dataloader:
    print("g.g",str(i))
    print(citing_tok_ids.shape)
    print(cited_tok_ids.shape)
    print(labels.shape)
    print(citing_cs_idx.shape)
    print(cited_cs_idx.shape)
    i = i+1
"""