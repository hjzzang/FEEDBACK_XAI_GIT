import pandas as pd
from nltk.tokenize import sent_tokenize
import os
import re
from itertools import product

dir = r'D:\2022_xai\data'
data_dir = dir+r'\hj_wisdomain'

final_pair_df= pd.read_json(dir+"//final_pair_df.json")
final_pair_onlyX_df= final_pair_df[final_pair_df["category"] == 'X'].reset_index(drop=True)
data_list= os.listdir(data_dir)

history_class_str = [' '.join(tuple(i)) for i in final_pair_onlyX_df["citing_history"].to_list()]
final_pair_onlyX_df["history_class"] = history_class_str
history_class = list(set(history_class_str))

history_dic = {'B8 B1 A1 A9':'case2', 'A9':'ing2', 'B1 A9 A1':'case2', 'A9 A1': 'ing2',
               'B9 B1 A1':'case2','B1 A9':'case2', 'B3 B8 B1 A1':'case1', 'B3 B1 A1':'case1',
               'B2 B1 A1':'case1', 'B8 B1 A1':'case1','B9 B1 A9 A1':'case2', 'B1 A1':'case1',
               'A1 A9': 'ing2', 'B1 A1 B2': 'case1', 'B9 B1 A1 A9':'case2', 'B2 B8 B1 A1':'case1', 'A1':'ing1', 'B1 A1 A9': 'case2' }

final_pair_onlyX_df["history_f"]= final_pair_onlyX_df["history_class"].map(history_dic)

citing_A1_id_list = [i[:2]+"0"+i[2:-2]+"A1" for i in final_pair_onlyX_df["citing_patent"]]
citing_A9_id_list = [i[:2]+"0"+i[2:-2]+"A9" for i in final_pair_onlyX_df["citing_patent"]]
citing_B1_id_list = [i[:2]+"0"+i[2:-2]+"B1" for i in final_pair_onlyX_df["citing_patent"]]
citing_B9_id_list = [i[:2]+"0"+i[2:-2]+"B9" for i in final_pair_onlyX_df["citing_patent"]]
cited_id_list = [i[:2]+"0"+i[2:] for i in final_pair_onlyX_df["cited_patent"]]

final_pair_onlyX_df["citing_A1"] = citing_A1_id_list
final_pair_onlyX_df["citing_A9"] = citing_A9_id_list
final_pair_onlyX_df["citing_B1"] = citing_B1_id_list
final_pair_onlyX_df["citing_B9"] = citing_B9_id_list
final_pair_onlyX_df["cited"]= cited_id_list