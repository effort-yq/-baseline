# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/28 21:45
# software: PyCharm

"""
文件说明：
    
"""

import json


def gen_new_trigger(tokens, trigger):
    start, end = 0, 0
    for i in range(len(tokens)):
        if trigger == tokens[i]:
            return trigger, [i, i + 1]

    for i in range(len(tokens)):
        if len(trigger) > len(tokens[i]):
            if trigger[0] in tokens[i]:
                start = i
            if trigger[-1] in tokens[i]:
                end = i+1
            if end != 0:
                return ''.join(tokens[start: end]), [start, end]

        elif len(trigger) == len(tokens[i]):
            if trigger[0] in tokens[i] and trigger[-1] in tokens[i]:
                start = i
                end = i + 1
                return ''.join(tokens[start: end]), [start, end]

            if trigger[0] in tokens[i]:
                start = i
            if trigger[-1] in tokens[i]:
                end = i + 1

            if end != 0:
                return ''.join(tokens[start: end]), [start, end]

        else:
            if trigger[0] in tokens[i] and trigger[-1] in tokens[i]:
                start = i
                end = i + 1
                return ''.join(tokens[start: end]), [start, end]
            if trigger[0] in tokens[i]:
                start = i

            if trigger[-1] in tokens[i]:
                end = i + 1

            if end != 0:
                return ''.join(tokens[start: end]), [start, end]


ent2id = {'Non-event': 0, 'Experiment': 1, 'Manoeuvre': 2, 'Deploy': 3,
             'Indemnity': 4, 'Support': 5, 'Accident': 6, 'Exhibit': 7}

data = []
with open('./baseline_test.json', 'r', encoding='utf-8') as f:
    examples = json.load(f)
    for idx, line in enumerate(examples):
        tokens = line['tokens']
        text = line['text']
        if len(line['entities']) == 0:
            data.append({'id': str(idx+1), 'event_mention': {}})
        else:
            event_type = line['entities'][0]['type']
            if event_type == 'Non-event':
                data.append({'id': str(idx+1), 'event_mention': {}})
            else:
                start_idx = line['entities'][0]['start_idx']
                end_idx = line['entities'][0]['end_idx']
                trigger = text[start_idx: end_idx+1]
                new_trigger, offset = gen_new_trigger(tokens, trigger)
                data.append({'id': str(idx+1), 'event_mention': {'event_type': line['entities'][0]['type'], 'trigger': {'text': new_trigger, 'offset': offset}}})

with open('./submit.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
