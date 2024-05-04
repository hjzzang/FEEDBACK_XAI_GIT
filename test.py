# Example of target with class indices
import torch
from torch import nn

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()

a = torch.randn(1,9)
b = torch.zeros(1,13)

max_lengths = [9,13]
citing_cs_maxlen_range = [i for i in range(max_lengths[0])]
cited_cs_maxlen_range = [i for i in range(max_lengths[1])]
pair_cs_range = []
for this_a in citing_cs_maxlen_range:
    for this_b in cited_cs_maxlen_range:
        pair_cs_range.append([this_a, this_b])

citing_cs_idx= [vl[0] for vl in pair_cs_range]
cited_cs_idx = [vl[1] for vl in pair_cs_range]

citing_cs_idx = torch.LongTensor(citing_cs_idx)
advice = torch.tensor([[[1.],
         [0.],
         [0.],
         [0.],
         [0.],
         [0.],
         [0.],
         [0.],
         [1.]]],)

advice = advice.flatten().tolist()
advice_list = [int(i + 1) for i in range(len(advice)) if advice[i] == 1]

advice = []
for hj in citing_cs_idx.flatten().tolist():
    if hj + 1 in advice_list:
        # increase the importance
        advice.append(1)
    else: advice.append(0)
advice = torch.LongTensor(advice)

import pandas as pd

dir = r'E:\OneDrive - dgu.ac.kr\0. DTILab\bithong\2022_XAI_MINI\data'
data_dir  = r'\pair_df_220718'
a = pd.read_json(dir+data_dir+"_train_test.json")
b = pd.read_json(dir+data_dir+"_dev_test.json")

c= a.append(b).reset_index(drop=True)
c.to_json(dir+data_dir+"_full_test.json")

