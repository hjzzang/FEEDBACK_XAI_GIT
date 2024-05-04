from typing import List

import numpy as np
import torch

def collate_input(batch: List[List[torch.Tensor]], max_len: int = None, fill_values: List[float] = None) -> \
    List[torch.Tensor]:

    lengths = np.array([[len(field_data) for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    #print("lengths.shape:", str(lengths.shape))
    #fill_values = [1,1, 0]
    fill_values = fill_values or [0.0] * num_fields
    max_lengths = lengths.max(axis=0)
    #print("max_lengths:", str(max_lengths))
    #max_size = [1, 200,200,1]
    #max_size = [512,512,1]
    max_size = [int(512/2),int(512/2),1]

    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len
    output = [torch.full([batch_size, max_lengths[field_idx], max_size[field_idx]],
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields)]

    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            for size_idx in range(len(max_size)):
                # seq_length
                data = batch[sample_idx][field_idx]
                try:output[field_idx][sample_idx][: data.shape[0]] = data
                except:
                    data = torch.reshape(data, (-1,1) )
                    output[field_idx][sample_idx][: data.shape[0]] = data

    citing_cs_maxlen_range = [i for i in range(max_lengths[0])]
    cited_cs_maxlen_range = [i for i in range(max_lengths[1])]
    pair_cs_range = []
    for this_a in citing_cs_maxlen_range:
        for this_b in cited_cs_maxlen_range:
            pair_cs_range.append([this_a, this_b])

    citing_cs_idx= [vl[0] for vl in pair_cs_range]
    cited_cs_idx = [vl[1] for vl in pair_cs_range]


    output.append(torch.LongTensor(citing_cs_idx))
    output.append(torch.LongTensor(cited_cs_idx))

    output.append(torch.LongTensor(max_lengths))


    #pair_id, citing_tok_ids, cited_tok_ids, label, citing_cs_idx, cited_cs_idx, max_lengths

    return output

