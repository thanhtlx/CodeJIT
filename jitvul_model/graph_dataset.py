import os
from torch.utils.data import DataLoader, Dataset
import torch
import random
import pandas as pd
import re


def preprocess_code_line(code):
    code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']',
                                                                                                                  ' ').replace(
        '.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')
    code = re.sub('``.*``', '<STR>', code)
    code = re.sub("'.*'", '<STR>', code)
    code = re.sub('".*"', '<STR>', code)
    code = re.sub('\d+', '<NUM>', code)
    code = code.split()
    code = ' '.join(code)
    return code.strip()


def hunk_empty(hunk):
    if hunk.strip() == '':
        return True


def get_hunk_from_diff(diff):
    hunk_list = []
    hunk = ''
    for line in diff.splitlines():
        if line.startswith(('+', '-')):
            hunk += line.strip() + '\n'
        else:
            if not hunk_empty(hunk):  # finish a hunk
                hunk = hunk[:-1]
                hunk_list.append(hunk)
                hunk = ''
    if not hunk_empty(hunk):
        hunk_list.append(hunk)
    return hunk_list


class Example(object):
    def __init__(self, commit_id, input_ids, attn, label):
        self.commit_id = commit_id
        self.input_ids = input_ids
        self.attn = attn
        self.label = label


class GraphDataset(Dataset):
    def __init__(self, data_path, tokenizer, _params):
        # csv midas okie
        self.examples = list()
        self.tokenizer = tokenizer
        df = pd.read_csv(data_path)
        cms = set(df['commit_id'])
        self.max_tokens = _params['MAX_TOKEN']
        self.max_hunks = _params['MAX_HUNKS']
        for cm in cms:
            tmp_df = df[df['commit_id'] == cm]
            hunks_cm = list()
            label = None
            for _, row in tmp_df.iterrows():
                diff = row['diff']
                label = row['label']
                hunks = get_hunk_from_diff(diff)
                for hunk in hunks:
                    hunk_tokens = list()
                    for line in hunk.splitlines():
                        if line.startswith('+'):
                            hunk_tokens += ["ADD"]
                        if line.startswith('-'):
                            hunk_tokens += ["DEL"]
                        line = preprocess_code_line(line[1:])
                        hunk_tokens += line.split()
                    hunks_cm.append(' '.join(hunk_tokens))
            # append(sample)
            # hunk_map[cm] = hunks_cm
            hunks_cm = hunks_cm[:self.max_hunks]
            ids_hunks = list()
            attn_hunks = list()
            for text in hunks_cm:
                encode = tokenizer.encode_plus(
                    text,
                    truncation=True,
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors='pt',
                )
                ids_hunks.append(encode.input_ids)
                attn_hunks.append(encode.input_ids)
            self.examples.append(Example(cm, torch.tensor(
                ids_hunks), torch.tensor(attn_hunks), torch.tensor([label], dtype = int)))

    def __getitem__(self, index):
        # self.commit_id = commit_id
        # self.input_ids = input_ids
        # self.attn = attn
        # self.label = label
        item = self.examples[index]
        return item.input_ids, item.attn, item.label, item.commit_id
    def __len__(self):
        return len(self.examples)

# only batch = 1


def collate_batch(batch):
    _data = batch[0]
    return _data
