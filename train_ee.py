# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/28 20:54
# software: PyCharm

"""
文件说明：
    
"""
from dataloader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import torch
from GlobalPointer import GlobalPointer, MetricsCalculator
from tqdm import tqdm
import json
import numpy as np

bert_model_path = r'E:\hugg_transformer\pre_model\chinese_wwm_ext_pytorch'
train_path = './train_ee.json'
eval_path = './valid_ee.json'
device = torch.device('cuda')

batch_size = 4

ENT_CLS_NUM = 8
ent2id, id2ent = {'Non-event': 0, 'Experiment': 1, 'Manoeuvre': 2, 'Deploy': 3,
             'Indemnity': 4, 'Support': 5, 'Accident': 6, 'Exhibit': 7}, {}
for k, v in ent2id.items(): id2ent[v] = k
#tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

train = EntDataset(load_data(train_path), tokenizer=tokenizer)
train_loader = DataLoader(train, batch_size=batch_size, collate_fn=train.collate, shuffle=True)
eval = EntDataset(load_data(eval_path), tokenizer=tokenizer)
eval_loader = DataLoader(eval, batch_size=batch_size, collate_fn=eval.collate, shuffle=True)

encoder = BertModel.from_pretrained(bert_model_path)
model = GlobalPointer(encoder, ENT_CLS_NUM, 64).to(device)  # 8个事件类型
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
global_steps = 0
for eo in range(10):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(train_loader):
        global_steps += 1
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device), labels.to(device)
        logits = model(input_ids, attention_mask, segment_ids)  # [batch_size, 8, seq_len, seq_len]
        loss = loss_fun(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        global_steps += 1
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        sample_f1 = metrics.get_sample_f1(logits, labels)
        total_loss += loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        print("trian_loss:", avg_loss, "\t train_f1:", avg_f1)

  
    with torch.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(eval_loader, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids)
            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = total_f1_ / (len(eval_loader))
        avg_precision = total_precision_ / (len(eval_loader))
        avg_recall = total_recall_ / (len(eval_loader))
        print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1, avg_precision, avg_recall))
    
        if avg_f1 > max_f:
            torch.save(model.state_dict(), './outputs/ent_model.pth'.format(eo))
            max_f = avg_f1
        model.train()