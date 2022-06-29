# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/28 21:12
# software: PyCharm

"""
文件说明：
    
"""
from transformers import BertModel, BertTokenizerFast
from GlobalPointer import GlobalPointer
import json
import torch
import numpy as np
from tqdm import tqdm
bert_model_path = r'E:\hugg_transformer\pre_model\chinese_wwm_ext_pytorch'
save_model_path = './outputs/ent_model.pth'
device = torch.device('cuda')

max_len = 150
ent2id, id2ent = {'Non-event': 0, 'Experiment': 1, 'Manoeuvre': 2, 'Deploy': 3,
             'Indemnity': 4, 'Support': 5, 'Accident': 6, 'Exhibit': 7}, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder =BertModel.from_pretrained(bert_model_path)
model = GlobalPointer(encoder, 8, 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda'))
model.eval()

def NER_RELATION(text, tokens, tokenizer, model, max_len=max_len):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1]-1])
    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).cuda()
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).cuda()
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).cuda()
    # score (batch_size, ent_type_size, seq_len, seq_len)  -->> (ent_type_size, seq_len, seq_len)
    scores = model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx": new_span[start][0], "end_idx": new_span[end][-1], "type": id2ent[l]})
    return {"text": text, "tokens": tokens, "entities": entities}

if __name__ == '__main__':
    all_ = []
    for d in tqdm(json.load(open('./test_ee.json', 'r', encoding='utf-8'))):
        all_.append(NER_RELATION(d["text"], d['tokens'], tokenizer=tokenizer, model=model))
    json.dump(
        all_,
        open('./baseline_test.json', 'w'),
        indent=4,
        ensure_ascii=False
    )