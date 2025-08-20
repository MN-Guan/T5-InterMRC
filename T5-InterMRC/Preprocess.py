import torch
import numpy as np

# In[5]:
# data:                   [5 4 3 2]        or  [6, 5, 3]
# max_len:                5
# decoder_input:          [0, 5, 4, 3, 2]  or  [0, 6, 5, 3, 0]
# decoder_attention_mask: [1, 1, 1, 1, 1]  or  [1, 1, 1, 1, 0]
# labels:                 [5, 4, 3, 2, 1]  or  [6, 5, 3, 1, -100]


def data_padding_input_ids(max_len, data):
    new_data = torch.zeros(max_len, dtype=torch.long, device=data.device)
    new_data[:data.shape[0]] = data
    return new_data

def data_padding_attention_mask(max_len, input_len, device):
    new_data = torch.zeros(max_len, dtype=torch.long, device=device)
    new_data[:input_len] = 1
    return new_data


def data_padding_decoder_input(max_len, data):
    new_data = torch.zeros(max_len, dtype=torch.long)
    data = torch.tensor(data, dtype=torch.long)
    new_data[1:len(data)+1] = data
    # the start idx of bart is 2. bart_decoder_input_ids: [2, 0, xxx, xxx, ...]
    if data[0] == 0:
        new_data[0] = 2
    return new_data

def data_padding_decoder_attention_mask(max_len, data):
    # bart_decoder_mask = [1, 1, 1, 1, 0, 0, ...]
    new_data = torch.zeros(max_len, dtype=torch.long)
    data = torch.tensor(data, dtype=torch.long)
    new_data[:len(data)+1] = 1
    return new_data

def data_padding_labels(max_len, data):
    # bart_label: [0, xxx, xxx, ...]
    new_data = torch.zeros(max_len, dtype=torch.long)
    data = torch.tensor(data, dtype=torch.long)
    new_data[:len(data)] = data
    new_data[len(data)] = 1
    new_data = torch.where(new_data==0, -100, new_data)
    if data[0] == 0:
        new_data[0] = 0
    return new_data

def mycollate(datasets):
    label_len_list = [len(example['labels']) for example in datasets]
    # add 1
    label_max_len = max(label_len_list)+1
    input_len_list = [len(example['input_ids']) for example in datasets]
    input_max_len = max(input_len_list)
    new_datasets = list()
    for example in datasets:
        new_datasets.append({'input_ids': data_padding_eva(input_max_len, torch.tensor(example['input_ids'])), 
                             'attention_mask': data_padding_eva(input_max_len, torch.tensor(example['attention_mask'])),
                             'decoder_input_ids': data_padding_decoder_input(label_max_len, example['labels']),
                             'decoder_attention_mask': data_padding_decoder_attention_mask(label_max_len, example['labels']),
                             'labels': data_padding_labels(label_max_len, example['labels'])})
    new_datasets2 = dict()
    new_datasets2['input_ids'] = torch.stack([example['input_ids'] for example in new_datasets], 0)
    new_datasets2['attention_mask'] = torch.stack([example['attention_mask'] for example in new_datasets], 0)
    new_datasets2['decoder_input_ids'] = torch.stack([example['decoder_input_ids'] for example in new_datasets], 0)
    new_datasets2['decoder_attention_mask'] = torch.stack([example['decoder_attention_mask'] for example in new_datasets], 0)
    new_datasets2['labels'] = torch.stack([example['labels'] for example in new_datasets], 0)

    return new_datasets2

def mycollate_RACE(datasets, entail_labels):
    label_len_list = [len(example['labels']) for example in datasets]
    # add 1
    label_max_len = max(label_len_list)+1
    input_len_list = [len(example['input_ids']) for example in datasets]
    input_max_len = max(input_len_list)
    new_datasets = list()
    for example in datasets:
        new_datasets.append({'input_ids': data_padding_eva(input_max_len, torch.tensor(example['input_ids'])), 
                             'attention_mask': data_padding_eva(input_max_len, torch.tensor(example['attention_mask'])),
                             'decoder_input_ids': data_padding_decoder_input(label_max_len, example['labels']),
                             'decoder_attention_mask': data_padding_decoder_attention_mask(label_max_len, example['labels']),
                             'reading_labels': data_padding_labels(label_max_len, example['labels']),
                             'entail_labels': torch.tensor(entail_labels, dtype=torch.long)})
    new_datasets2 = dict()
    new_datasets2['input_ids'] = torch.stack([example['input_ids'] for example in new_datasets], 0)
    new_datasets2['attention_mask'] = torch.stack([example['attention_mask'] for example in new_datasets], 0)
    new_datasets2['decoder_input_ids'] = torch.stack([example['decoder_input_ids'] for example in new_datasets], 0)
    new_datasets2['decoder_attention_mask'] = torch.stack([example['decoder_attention_mask'] for example in new_datasets], 0)
    new_datasets2['reading_labels'] = torch.stack([example['reading_labels'] for example in new_datasets], 0)
    new_datasets2['entail_labels'] = torch.stack([example['entail_labels'] for example in new_datasets], 0)

    return new_datasets2

def mycollate_mrc(datasets):
    re_label_len_list = [len(example['re_labels']) for example in datasets]
    qa_label_len_list = [len(example['qa_labels']) for example in datasets]
    # add 1
    re_label_max_len = max(re_label_len_list)+1
    qa_label_max_len = max(qa_label_len_list)+1
    input_len_list = [len(example['input_ids']) for example in datasets]
    input_max_len = max(input_len_list)
    new_datasets = list()
    for example in datasets:
        new_datasets.append({'input_ids': data_padding_eva(input_max_len, torch.tensor(example['input_ids'])), 
                             'attention_mask': data_padding_eva(input_max_len, torch.tensor(example['attention_mask'])),
                             're_decoder_input_ids': data_padding_decoder_input(re_label_max_len, example['re_labels']),
                             're_decoder_attention_mask': data_padding_decoder_attention_mask(re_label_max_len, example['re_labels']),
                             're_labels': data_padding_labels(re_label_max_len, example['re_labels']),
                             'qa_decoder_input_ids': data_padding_decoder_input(qa_label_max_len, example['qa_labels']),
                             'qa_decoder_attention_mask': data_padding_decoder_attention_mask(qa_label_max_len, example['qa_labels']),
                             'qa_labels': data_padding_labels(qa_label_max_len, example['qa_labels'])})
    new_datasets2 = dict()
    new_datasets2['input_ids'] = torch.stack([example['input_ids'] for example in new_datasets], 0)
    new_datasets2['attention_mask'] = torch.stack([example['attention_mask'] for example in new_datasets], 0)
    new_datasets2['re_decoder_input_ids'] = torch.stack([example['re_decoder_input_ids'] for example in new_datasets], 0)
    new_datasets2['re_decoder_attention_mask'] = torch.stack([example['re_decoder_attention_mask'] for example in new_datasets], 0)
    new_datasets2['re_labels'] = torch.stack([example['re_labels'] for example in new_datasets], 0)
    new_datasets2['qa_decoder_input_ids'] = torch.stack([example['qa_decoder_input_ids'] for example in new_datasets], 0)
    new_datasets2['qa_decoder_attention_mask'] = torch.stack([example['qa_decoder_attention_mask'] for example in new_datasets], 0)
    new_datasets2['qa_labels'] = torch.stack([example['qa_labels'] for example in new_datasets], 0)

    return new_datasets2

def data_padding_eva(max_len, data):
    new_data = torch.zeros(max_len, dtype=torch.long)
    data = torch.tensor(data, dtype=torch.long)
    new_data[:len(data)] = data
    return new_data

def mycollate_eva(datasets):
    len_list = [len(example['input_ids']) for example in datasets]
    max_len = max(len_list)
    new_datasets = list()
    for example in datasets:
        new_datasets.append({'input_ids': data_padding_eva(max_len, example['input_ids']), 
                             'attention_mask': data_padding_eva(max_len, example['attention_mask'])})
    new_datasets2 = dict()
    new_datasets2['input_ids'] = torch.stack([example['input_ids'] for example in new_datasets], 0)
    new_datasets2['attention_mask'] = torch.stack([example['attention_mask'] for example in new_datasets], 0)
    return new_datasets2

def random_extract(a, b):
    total = len(a) + len(b)
    mapping = {0:a, 1:b}
    c = []
    p = np.array([len(a)/total, len(b)/total])
    for i in range(total):
        index = np.random.choice([0, 1], p=p.ravel())
        c.append(mapping[index].pop())
        if a == []:
            c.extend(b)
            break
        elif b == []:
            c.extend(a)
            break
    return c

