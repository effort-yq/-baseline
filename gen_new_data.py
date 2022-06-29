# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/27 15:51
# software: PyCharm

"""
文件说明：
    
"""

import json
train_data = json.load(open('./train_7000.json', 'r', encoding='utf-8'))
val_data = json.load(open('./valid_1500.json', 'r', encoding='utf-8'))
test_data = json.load(open('./test_dataset_A.json', 'r', encoding='utf-8'))

train_new = []
for line in train_data:
    tmp_sent = line['sentence']
    tokens = line['tokens']
    if line['event_mention']:
        event_type = line['event_mention']['event_type']
        trigger = line['event_mention']['trigger']['text']
        offset = line['event_mention']['trigger']['offset'][-1]
        new_sent = ''.join(tokens[:offset])
        start_idx = new_sent.index(trigger[0])
        end_idx = new_sent.index(trigger[-1])
        dic = {'text': tmp_sent, 'entities': [{'start_idx': start_idx, 'end_idx': end_idx, 'type': event_type, 'entity': trigger}]}
    else:
        dic = {'text': tmp_sent, 'entities': [{'start_idx': 0, 'end_idx': 0, 'type': 'Non-event', 'entity': ''}]}

    train_new.append(dic)

with open('train_ee.json', 'w', encoding='utf-8') as w:
    json.dump(train_new, w, ensure_ascii=False, indent=4)


valid_new = []
for line in val_data:
    tmp_sent = line['sentence']
    tokens = line['tokens']
    if line['event_mention']:
        event_type = line['event_mention']['event_type']
        trigger = line['event_mention']['trigger']['text']
        offset = line['event_mention']['trigger']['offset'][-1]
        new_sent = ''.join(tokens[:offset])
        start_idx = new_sent.index(trigger[0])
        end_idx = new_sent.index(trigger[-1])
        dic = {'text': tmp_sent, 'tokens': tokens, 'entities': [{'start_idx': start_idx, 'end_idx': end_idx, 'type': event_type,
                                               'entity': trigger, 'offset': line['event_mention']['trigger']['offset']}]}
    else:
        dic = {'text': tmp_sent, 'tokens': tokens, 'entities': [{'start_idx': 0, 'end_idx': 0, 'type': 'Non-event', 'entity': '', 'offset': []}]}

    valid_new.append(dic)

with open('valid_ee.json', 'w', encoding='utf-8') as w:
    json.dump(valid_new, w, ensure_ascii=False, indent=4)


test_new = []
for line in test_data:
    tmp_sent = line['sentence']
    tokens = line['tokens']
    dic = {'text': tmp_sent, 'tokens': tokens, 'entities': []}
    test_new.append(dic)

with open('test_ee.json', 'w', encoding='utf-8') as w:
    json.dump(test_new, w, ensure_ascii=False, indent=4)