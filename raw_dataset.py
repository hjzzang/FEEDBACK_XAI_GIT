import pandas as pd
from nltk.tokenize import sent_tokenize
import os
import re

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

citing_A1_raw_patent_df= pd.DataFrame()
citing_A9_raw_patent_df= pd.DataFrame()
citing_B1_raw_patent_df= pd.DataFrame()
citing_B9_raw_patent_df= pd.DataFrame()
cited_raw_patent_df= pd.DataFrame()

def claim_split(claim):
    if type(claim) == str:
        claim = claim.replace('Claim', 'claim')
        split_claim = [i for i in sent_tokenize(claim) if len(i) > 10]
        if len(split_claim) == 1:
            dot_error = re.findall('[\\.[A-Z]+', claim)
            if len(dot_error) != 0:
                for error in dot_error:
                    error_start = claim.find(error)
                    claim = claim[:error_start + 1] + " " + claim[error_start + 1:]
            split_claim = [i for i in sent_tokenize(claim) if len(i) > 10]

    elif type(claim) == list:
        split_claim = claim
    return split_claim

def claim_mask(claim_len):
    mask_dict = {True:1, False:0}
    claim_mask_list = []
    for i in range(claim_len):
        this_mask_list = [mask_dict[i == j] for j in range(claim_len)]
        claim_mask_list.append(this_mask_list)
    return claim_mask_list

def feedback_mask(claim_srprt, claim_len):
    claim_srprt_id_int = [int(i) for i in claim_srprt]
    feedback_mask_list = []
    for i in range(int(claim_len)):
        if i+1 in claim_srprt_id_int:
            feedback_mask_list.append(1)
        else:
            feedback_mask_list.append(0)
    return feedback_mask_list


max_claim_com_len = 0
for i, this_file in enumerate(data_list):
    if i% 100 == 0: print(str(i),"/",str(len(data_list)))
    #print("len of cited:", str(len(cited_raw_patent_df)))
    #load downloaded patent raw file (from wisdomain)
    #this_file = data_list[i]
    data_df = pd.read_csv(data_dir+"//"+this_file, header=4)

    this_citing_A1_df = data_df[(data_df["번호"].isin(citing_A1_id_list))].reset_index(drop=True)
    if len(this_citing_A1_df)>0 :
        this_citing_A1_df["citing_A1"] = this_citing_A1_df[["번호"]]
        this_citing_A1_df = this_citing_A1_df[["citing_A1", "전체 청구항", "국제특허분류"]]
        this_citing_A1_df_ = this_citing_A1_df[["citing_A1"]]
        this_citing_A1_df_["citing_A1_ipc"] = this_citing_A1_df[["국제특허분류"]]

        this_citng_split_claim = [claim_split(str(this_citing_A1_df.iloc[i]["전체 청구항"])) for i in range(len(this_citing_A1_df))]
        this_claim_len = [len(this_citng_split_claim[i]) for i in range(len(this_citing_A1_df))]
        this_citing_A1_df_["claim_A1_txt"] = this_citng_split_claim
        this_citing_A1_df_["claim_A1_len"] = this_claim_len
        this_citing_A1_df_["claim_A1_mask"] = [claim_mask(i) for i in this_claim_len]
        citing_A1_raw_patent_df = citing_A1_raw_patent_df.append(this_citing_A1_df_).drop_duplicates(
            'citing_A1').reset_index(drop=True)

    this_citing_A9_df = data_df[(data_df["번호"].isin(citing_A9_id_list))].reset_index(drop=True)
    if len(this_citing_A9_df)>0:
        this_citing_A9_df["citing_A9"] = this_citing_A9_df[["번호"]]
        this_citing_A9_df = this_citing_A9_df[["citing_A9", "전체 청구항", "국제특허분류"]]
        this_citing_A9_df_ = this_citing_A9_df[["citing_A9"]]
        this_citing_A9_df_["citing_A9_ipc"] = this_citing_A9_df[["국제특허분류"]]

        this_citng_split_claim = [claim_split(str(this_citing_A9_df.iloc[i]["전체 청구항"])) for i in range(len(this_citing_A9_df))]
        this_claim_len = [len(this_citng_split_claim[i]) for i in range(len(this_citing_A9_df))]
        this_citing_A9_df_["claim_A9_txt"] = this_citng_split_claim
        this_citing_A9_df_["claim_A9_len"] = this_claim_len
        this_citing_A9_df_["claim_A9_mask"] = [claim_mask(i) for i in this_claim_len]
        citing_A9_raw_patent_df = citing_A9_raw_patent_df.append(this_citing_A9_df_).drop_duplicates(
            'citing_A9').reset_index(drop=True)

    this_citing_B1_df = data_df[(data_df["번호"].isin(citing_B1_id_list))].reset_index(drop=True)
    if len(this_citing_B1_df)>0:
        this_citing_B1_df["citing_B1"] = this_citing_B1_df[["번호"]]
        this_citing_B1_df = this_citing_B1_df[["citing_B1", "전체 청구항", "국제특허분류"]]
        this_citing_B1_df_ = this_citing_B1_df[["citing_B1"]]
        this_citing_B1_df_["citing_B1_ipc"] = this_citing_B1_df[["국제특허분류"]]

        this_citng_split_claim = [claim_split(str(this_citing_B1_df.iloc[i]["전체 청구항"])) for i in range(len(this_citing_B1_df))]
        this_claim_len = [len(this_citng_split_claim[i]) for i in range(len(this_citing_B1_df))]
        this_citing_B1_df_["claim_B1_txt"] = this_citng_split_claim
        this_citing_B1_df_["claim_B1_len"] = this_claim_len
        this_citing_B1_df_["claim_B1_mask"] = [claim_mask(i) for i in this_claim_len]
        citing_B1_raw_patent_df = citing_B1_raw_patent_df.append(this_citing_B1_df_).drop_duplicates(
            'citing_B1').reset_index(drop=True)

    this_citing_B9_df = data_df[(data_df["번호"].isin(citing_B9_id_list))].reset_index(drop=True)
    if len(this_citing_B9_df)>0:
        this_citing_B9_df["citing_B9"] = this_citing_B9_df[["번호"]]
        this_citing_B9_df = this_citing_B9_df[["citing_B9", "전체 청구항", "국제특허분류"]]
        this_citing_B9_df_ = this_citing_B9_df[["citing_B9"]]
        this_citing_B9_df_["citing_B9_ipc"] = this_citing_B9_df[["국제특허분류"]]

        this_citng_split_claim = [claim_split(str(this_citing_B9_df.iloc[i]["전체 청구항"])) for i in range(len(this_citing_B9_df))]
        this_claim_len = [len(this_citng_split_claim[i]) for i in range(len(this_citing_B9_df))]
        this_citing_B9_df_["claim_B9_txt"] = this_citng_split_claim
        this_citing_B9_df_["claim_B9_len"] = this_claim_len
        this_citing_B9_df_["claim_B9_mask"] = [claim_mask(i) for i in this_claim_len]
        citing_B9_raw_patent_df = citing_B9_raw_patent_df.append(this_citing_B9_df_).drop_duplicates(
            'citing_B9').reset_index(drop=True)

    this_cited_df = data_df[(data_df["번호"].isin(cited_id_list))].reset_index(drop=True)
    if len(this_cited_df)>0:
        this_cited_df["cited"] = this_cited_df[["번호"]]
        this_cited_df = this_cited_df[["cited", "전체 청구항", "국제특허분류"]]
        this_cited_df_ = this_cited_df[["cited"]]
        this_cited_df_["cited_ipc"] = this_cited_df[["국제특허분류"]]

        this_cited_split_claim = [claim_split(str(this_cited_df.iloc[i]["전체 청구항"])) for i in range(len(this_cited_df))]
        this_claim_len = [len(this_cited_split_claim[i]) for i in range(len(this_cited_df))]
        this_cited_df_["claim_cited_txt"] = this_cited_split_claim
        this_cited_df_["claim_cited_len"] = this_claim_len
        this_cited_df_["claim_cited_mask"] = [claim_mask(i) for i in this_claim_len]
        cited_raw_patent_df = cited_raw_patent_df.append(this_cited_df_).drop_duplicates(
            'cited').reset_index(drop=True)

final_pair_m = final_pair_onlyX_df.merge(citing_A1_raw_patent_df, how='left', on='citing_A1' ).reset_index(drop=True)
final_pair_m = final_pair_m.merge(citing_A9_raw_patent_df, how='left', on='citing_A9' ).reset_index(drop=True)
final_pair_m = final_pair_m.merge(citing_B1_raw_patent_df, how='left', on='citing_B1' ).reset_index(drop=True)
final_pair_m = final_pair_m.merge(citing_B9_raw_patent_df, how='left', on='citing_B9' ).reset_index(drop=True)
final_pair_m = final_pair_m.merge(cited_raw_patent_df, how='left', on='cited' ).reset_index(drop=True)

final_pair_m  = final_pair_m.loc[final_pair_m.astype(str).drop_duplicates().index].reset_index(drop=True)
final_pair_m = final_pair_m.fillna("")

#dataset w/o feedback
index = []
pair_srprt = []
label = []
case = []

for hj in range(len(final_pair_m)):
    if hj % 1000 == 0: print(hj)
    if ((final_pair_m.iloc[hj]["history_f"] == "case1") and (len(final_pair_m.iloc[hj]["claim_B1_txt"])>0)and (len(final_pair_m.iloc[hj]["claim_A1_txt"])>0)):
        try:
            if (len(final_pair_m.iloc[hj]['claim_B1_txt']) != final_pair_m.iloc[hj]['claim_A1_txt'] and (len(final_pair_m.iloc[hj]["claim_A1_txt"])+len(final_pair_m.iloc[hj]["claim_cited_txt"])<33)):
                index.append(str(hj))
                pair_srprt.append([final_pair_m.iloc[hj]['claim_A1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
                label.append(1)
                case.append("case1")

            if ((len(final_pair_m.iloc[hj]["claim_B1_txt"])+len(final_pair_m.iloc[hj]["claim_cited_txt"])<33)):
                index.append(str(hj))
                pair_srprt.append([final_pair_m.iloc[hj]['claim_B1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
                label.append(1)
                case.append("case1")

                index.append(str(hj))
                pair_srprt.append([final_pair_m.iloc[hj]['citing_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
                label.append(0)
                case.append("case1")
        except:
            print(hj)
    if ((final_pair_m.iloc[hj]["history_f"] == "case2") and (len(final_pair_m.iloc[hj]["claim_A1_txt"])>0) and (len(final_pair_m.iloc[hj]["claim_B1_txt"])>0)
    and (len(final_pair_m.iloc[hj]["claim_A1_txt"])+len(final_pair_m.iloc[hj]["claim_cited_txt"])<33) and (len(final_pair_m.iloc[hj]["claim_B1_txt"])+len(final_pair_m.iloc[hj]["claim_cited_txt"])<33)):
        try:
            index.append(str(hj))
            pair_srprt.append([final_pair_m.iloc[hj]['claim_A1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
            label.append(0)
            case.append("case2")

            index.append(str(hj))
            pair_srprt.append([final_pair_m.iloc[hj]['claim_B1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
            label.append(1)
            case.append("case2")

            index.append(str(hj))
            pair_srprt.append([final_pair_m.iloc[hj]['citing_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
            label.append(0)
            case.append("case2")
        except:
            print(hj)
    if ((hj<10005) and (final_pair_m.iloc[hj]["history_f"] == "ing1") and (len(final_pair_m.iloc[hj]["claim_A1_txt"])>0)
    and (len(final_pair_m.iloc[hj]["claim_A1_txt"])+len(final_pair_m.iloc[hj]["claim_cited_txt"])<33)):
        try:
            index.append(str(hj))
            pair_srprt.append([final_pair_m.iloc[hj]['claim_A1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
            label.append(0)
            case.append("ing1")
        except:
            print(hj)
final_df = pd.DataFrame({"label":label, "pair_srprt": pair_srprt})
final_df_ = pd.DataFrame({"index":index, "case":case, "label":label, "pair_srprt": pair_srprt})

final_df.to_csv(dir+'//pair_final_df_txt.txt', sep = '\t', index = False, header=None, mode='a', encoding='utf-8')
final_df.to_json(dir+'//pair_final_df_txt.json', orient='records')
final_df_.to_json(dir+'//pair_final_df_w_idtxt.json', orient='records')

final_len = len(final_df)
train_result = final_df.loc[:int(final_len*0.6)]
dev_result = final_df.loc[int(final_len*0.6)+1:int(final_len*0.9)].reset_index(drop=True)
test_result = final_df.loc[int(final_len*0.9)+1:int(final_len)].reset_index(drop=True)


#result.to_csv(dir+'//pair_final_df_depth3_hj.txt', sep = '\t', index = False, header=None, mode='a', encoding='utf-8')
save_dir = r'D:\FEEDBACK_XAI\data'
train_result.to_json(save_dir+'//pair_df_train.json', orient='records')
train_result[:100].to_json(save_dir+'//pair_df_small_train.json', orient='records')
dev_result.to_json(save_dir+'//pair_df_dev.json', orient='records')
test_result.to_json(save_dir+'//pair_df_test.json', orient='records')

#dataset with feedback
#spacy download en_core_web_sm
import spacy
nlp = spacy.load('en_core_web_sm')
def similarityscore(text1, text2 ):
    doc1 = nlp( text1 )
    doc2 = nlp( text2 )
    similarity = doc1.similarity( doc2 )
    return similarity

index = []
pair_srprt = []
label = []
feedback_list = []

#feedback
for hj in range(len(final_pair_m)):
    if hj % 100 == 0: print(hj)
    if ((final_pair_m.iloc[hj]["history_f"] == "case1") and (len(final_pair_m.iloc[hj]["claim_A1_txt"])>0) and (len(final_pair_m.iloc[hj]["claim_B1_txt"])>0)):
        try:
            this_sim = similarityscore(' '.join(final_pair_m.iloc[hj]["claim_A1_txt"]),' '.join(final_pair_m.iloc[hj]["claim_B1_txt"]))
            this_feedback = feedback_mask(final_pair_m.loc[hj]["citing_id"], final_pair_m.loc[hj]['claim_A1_len'])
            index.append(str(hj))
            pair_srprt.append([final_pair_m.iloc[hj]['claim_A1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
            label.append(1)
            feedback_list.append(this_feedback)
        except:
            print(hj)
    if ((hj<10005) and (final_pair_m.iloc[hj]["history_f"] == "ing1")):
        try:
            this_sim = similarityscore(' '.join(final_pair_m.iloc[hj]["claim_A1_txt"]),' '.join(final_pair_m.iloc[hj]["claim_B1_txt"]))
            this_feedback = feedback_mask(final_pair_m.loc[hj]["claim_A1_txt"], final_pair_m.loc[hj]['claim_A1_len'])
            index.append(str(hj))
            pair_srprt.append([final_pair_m.iloc[hj]['claim_A1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
            label.append(0)
            feedback_list.append(this_feedback)
        except:
            print(hj)
    if ((final_pair_m.iloc[hj]["history_f"] == "case2") and (len(final_pair_m.iloc[hj]["claim_A1_txt"])>0) and (len(final_pair_m.iloc[hj]["claim_B1_txt"])>0)):
        try:
            index.append(str(hj))
            pair_srprt.append([final_pair_m.iloc[hj]['claim_A1_txt'], final_pair_m.iloc[hj]['claim_cited_txt']])
            label.append(1)
            this_feedback = feedback_mask(final_pair_m.loc[hj]["claim_A1_txt"], final_pair_m.loc[hj]['claim_A1_len'])
            feedback_list.append(this_feedback)
        except:
            print(hj)

final_w_fb_df = pd.DataFrame({"index":index, "pair_srprt": pair_srprt, "feedback":feedback_list})



