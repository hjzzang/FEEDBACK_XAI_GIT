import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pandas as pd


class EPfulldataset_w_adv(Dataset):
    def __init__(self, directory, prefix, bert_path, max_length: int = int(512/2)):
        super().__init__()
        self.max_length = max_length
        # self.max_cc_len = 7676
        srprts = pd.read_json(directory + prefix + '.json', orient='records')

        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)
        self.result = []
        for hj in range(len(srprts)):
            line = srprts.iloc[hj]
            self.result.append((line['label'], line['pair_srprt'], line['feedback']))

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        # pair_id, pair_srprt, label = self.result[idx]
        label, pair_srprt, advice = self.result[idx]

        citing_tok_ids = torch.FloatTensor(
            [self.tokenizer.encode(i, add_special_tokens=False, padding='max_length', max_length=self.max_length,
                                   truncation=True) for i in pair_srprt[0]])
        cited_tok_ids = torch.FloatTensor(
            [self.tokenizer.encode(i, add_special_tokens=False, padding='max_length', max_length=self.max_length,
                                   truncation=True) for i in pair_srprt[1]])


        return citing_tok_ids, cited_tok_ids, torch.LongTensor([label]), torch.LongTensor(advice)