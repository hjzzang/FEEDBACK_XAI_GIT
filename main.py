#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : trainer.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/16 21:55
@version: 1.0
@desc  :
"""

import argparse
import json
import os
from functools import partial
# partial gives default values to the parameters of a function that would otherwise not have default values.

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
import pandas as pd

from EP_full_collate_functions import collate_input
from EP_full_collate_functions_w_adv import collate_input_w_adv
from EP_full_model import ExplainableModel
from EP_full_dataset import EPfulldataset
from EP_full_dataset_w_adv import EPfulldataset_w_adv

class ExplainNLP(pl.LightningModule):

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.model = ExplainableModel(self.bert_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_dir)
        self.loss_fn = CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.output = []
        self.check_data = []

        self.output_df = pd.DataFrame()

        self.hjhj = 0

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_dataloader()) // self.args.accumulate_grad_batches * self.args.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, citing_tok_ids, cited_tok_ids, citing_cs_idx, cited_cs_idx):
        return self.model(citing_tok_ids, cited_tok_ids, citing_cs_idx, cited_cs_idx)

    def forward_w_adv(self, citing_tok_ids, cited_tok_ids, advice, citing_cs_idx, cited_cs_idx):
        return self.model(citing_tok_ids, cited_tok_ids, advice, citing_cs_idx, cited_cs_idx)

    def compute_loss_and_acc(self, batch, mode='train'):
        if self.args.adv == 'w/o':
            # pair_id, citing_tok_ids, cited_tok_ids, labels, citing_cs_idx, cited_cs_idx = batch
            citing_tok_ids, cited_tok_ids, labels, citing_cs_idx, cited_cs_idx, max_lengths = batch
            y_hat, a_ij = self.forward(citing_tok_ids, cited_tok_ids, citing_cs_idx, cited_cs_idx)
            # y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = labels.view(-1)

            # compute loss
            ce_loss = self.loss_fn(y_hat, y.long())
            # print("ce_loss:",ce_loss)
            # ce_loss = loss_fn(y_hat, y.long())
            reg_loss = self.args.lamb * a_ij.pow(2).sum(dim=1).mean()
            # print("reg_loss:", reg_loss)

        if self.args.adv == 'w':
            citing_tok_ids, cited_tok_ids, labels, advice, citing_cs_idx, cited_cs_idx, max_lengths = batch
            y_hat, a_ij = self.forward(citing_tok_ids, cited_tok_ids, citing_cs_idx, cited_cs_idx)

            advice_re = advice.flatten().tolist()
            advice_list = [int(i + 1) for i in range(len(advice_re)) if advice_re[i] == 1]

            advice_pair = []
            for hj in citing_cs_idx.flatten().tolist():
                if hj + 1 in advice_list:
                    # increase the importance
                    advice_pair.append(2)
                else: advice_pair.append(1)
            advice_pair = torch.LongTensor(advice_pair).cuda()

            y = labels.view(-1)

            # compute loss
            ce_loss = self.loss_fn(y_hat, y.long())
            # print("ce_loss:",ce_loss)
            # ce_loss = loss_fn(y_hat, y.long())
            multi_adv_a_ij = a_ij * advice_pair  # torch.Size([1, 84])
            reg_sum_w_1 = torch.div(multi_adv_a_ij, multi_adv_a_ij.sum())
            reg_loss = self.args.lamb * reg_sum_w_1.pow(2).sum(dim=1).mean()
            # print("reg_loss:", reg_loss)
        loss = ce_loss - reg_loss
        # print("loss:", loss)
        # compute acc
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)

        if mode == 'train':
            #print("train mode")
            acc = self.train_acc(predict_labels, y)

        else:
            acc = self.valid_acc(predict_labels, y)

        # if test, save extract spans

        if mode == 'test':

            #pair_doc_id = self.id_df.loc[self.id_df['id'] == pair_id]['doc_id'].values[0]
            try:
                values, indices = torch.topk(a_ij,self.args.span_topk)
                values = values.tolist()
                indices = indices.tolist()


                output_score = []
                output_top_k = []
                output_top_k_txt = []

                for i in range(len(values)):

                    #print("citing_tok_ids.shape:",str(citing_tok_ids.shape))
                    #print("len of values:", str(len(values)))
                    this_citing_tok_ids = citing_tok_ids[i]
                    this_cited_tok_ids = cited_tok_ids[i]
                    #self.output.append("Result:", str(pair_doc_id))

                    this_score_list = []
                    this_top_k_list = []
                    this_top_k_txt_list = []

                    self.check_data.append(str(y[i].item()) + '-' + str(predict_labels[i].item()))
                    for j, cs_idx in enumerate(indices[i]):
                        topk_citing_cs_id = citing_cs_idx[cs_idx]
                        topk_cited_cs_id = cited_cs_idx[cs_idx]
                        topk_citing_id_list = this_citing_tok_ids[topk_citing_cs_id]
                        topk_cited_id_list = this_cited_tok_ids[topk_cited_cs_id]
                        self.output.append("_cs_id:"+str(topk_citing_cs_id.item()) )
                        self.output.append(
                            str(y[i].item()) + '<->' + str(predict_labels[i].item()) + '/ related claim pair num.:' + str(
                                citing_cs_idx[cs_idx].tolist()) + "-" + str(cited_cs_idx[cs_idx].tolist()))
                        self.output.append('\n')
                        score = values[i][j]
                        origin_citing_sentence = self.tokenizer.decode(topk_citing_id_list, skip_special_tokens=True)
                        origin_cited_sentence = self.tokenizer.decode(topk_cited_id_list, skip_special_tokens=True)
                        self.output.append(format('%.4f' % score) + '\n')
                        self.output.append(str("target:") + origin_citing_sentence + '\n')
                        self.output.append(str("prior:") + origin_cited_sentence + '\n')
                        self.output.append('\n')

                        this_score_list.append(format('%.4f' % score))
                        this_top_k_pair = str(topk_citing_cs_id.item())+"-"+str(topk_cited_cs_id.item())
                        this_top_k_list.append(this_top_k_pair)
                        this_top_k_txt_pair = origin_citing_sentence+"-"+origin_cited_sentence
                        this_top_k_txt_list.append(this_top_k_txt_pair)

                    output_score.append(this_score_list)
                    output_top_k.append(this_top_k_list)
                    output_top_k_txt.append(this_top_k_txt_list)
                    #print(max_lengths.tolist())

                    this_output_df = pd.DataFrame({"ID":self.hjhj,"Pred":str(y.item()), "Act":predict_labels.item(), "F_value":torch.Tensor.tolist(predict_scores.flatten())[predict_labels[i].item()],"max_length": [max_lengths.tolist()], "Score":[output_score], "TopK":[output_top_k], "Text":[output_top_k_txt]})
                    self.hjhj += 1
                    self.output_df = self.output_df.append(this_output_df).reset_index(drop=True)
                    self.output.append('\n')
            except:
                #print("error:",str(self.hjhj))
                self.hjhj += 1
        return loss, acc

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.valid_acc.compute()
        self.log('valid_acc_end', self.valid_acc.compute())

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss)
        return loss

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch, mode='dev')
        self.log('valid_acc', acc, on_step=False, on_epoch=True)
        self.log('valid_loss', loss)
        return loss

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        if self.args.adv == "w/o":
            dataset = EPfulldataset(directory=self.args.data_dir, prefix=prefix, bert_path=self.bert_dir)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.workers,
                collate_fn=partial(collate_input, fill_values=[1,1,0]),
                shuffle=False,
                drop_last=False
            )
        elif self.args.adv == "w":
            dataset = EPfulldataset_w_adv(directory=self.args.data_dir, prefix=prefix, bert_path=self.bert_dir)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.workers,
                collate_fn=partial(collate_input_w_adv, fill_values=[1, 1, 0, 0]),
                shuffle=False,
                drop_last=False
            )
        return dataloader

    def test_dataloader(self):
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch, mode='test')
        return {'test_loss': loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        with open(os.path.join(self.args.save_path, 'output.txt'), 'w', encoding='utf8') as f:
            f.writelines(self.output)
        with open(os.path.join(self.args.save_path, 'test.txt'), 'w', encoding='utf8') as f:
            f.writelines(self.check_data)
        self.output_df.to_json(self.args.save_path+"/output_df.json", orient='records')
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    #parser.add_argument("--max_length", default=int(512), type=int, help="max length of dataset")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="path to save checkpoints")
    parser.add_argument("--save_topk", default=500, type=int, help="save topk checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="checkpoint path on test step", default='./checkpoints/last.ckpt')
    parser.add_argument("--source_model_path", type=str, help="checkpoint path on test step")
    parser.add_argument("--span_topk", type=int, default=50, help="save topk spans on test step")
    parser.add_argument("--lamb", default=1.0, type=float, help="regularizer lambda")
    #parser.add_argument("--task", default='sst5', type=str, help="nlp tasks")
    parser.add_argument("--mode", default='train', type=str, help="either train or eval")
    parser.add_argument('--resume-from')
    parser.add_argument("--adv", default='w/o', type=str)
    return parser


def train(args):
    # if save path does not exits, create it
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    model = ExplainNLP(args)

    #if args.adv == "w":

    if args.source_model_path is not None:
        checkpoint = torch.load(args.source_model_path)
        model.load_state_dict(checkpoint['state_dict'])

    checkpoint_callback = ModelCheckpoint(
        #filepath=os.path.join(args.save_path, '{epoch}-{valid_loss:.4f}-{valid_acc_end:.4f}'),
        dirpath=os.path.join(args.save_path),
        filename='{epoch}-{train_loss:.4f}-{train_acc:.4f}-{valid_loss:.4f}-{valid_acc_end:.4f}',
        save_top_k=args.save_topk,
        save_last=True,
        monitor="valid_acc_end",
        mode="max",
    )
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )

    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        args_dict = args.__dict__
        del args_dict['tpu_cores']
        json.dump(args_dict, f, indent=4)
    trainer = Trainer.from_argparse_args(args,
                                         # checkpoint_callback=checkpoint_callback,
                                         callbacks=checkpoint_callback,
                                         # distributed_backend="ddp",
                                         logger=logger, num_sanity_val_steps=0)  # , num_sanity_val_steps=1
    trainer.fit(model)



def evaluate(args):
    model = ExplainNLP(args)
    #checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    trainer = Trainer.from_argparse_args(args)
    trainer.test(model)


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    else:
        raise Exception("unexpected mode!!!")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
